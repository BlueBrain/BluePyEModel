"""FitnessCalculatorConfiguration"""
import logging
from copy import deepcopy

from bluepyemodel.evaluation.efeature_configuration import EFeatureConfiguration
from bluepyemodel.evaluation.evaluator import LEGACY_PRE_PROTOCOLS
from bluepyemodel.evaluation.evaluator import PRE_PROTOCOLS
from bluepyemodel.evaluation.evaluator import seclist_to_sec
from bluepyemodel.evaluation.protocol_configuration import ProtocolConfiguration

logger = logging.getLogger(__name__)


def _get_apical_point(cell):
    """Get the apical point isec usign automatic apical point detection."""

    from morph_tool import apical_point
    from morph_tool import nrnhines
    from morphio import Morphology

    point = apical_point.apical_point_position(Morphology(cell.morphology.morphology_path))

    return nrnhines.point_to_section_end(cell.icell.apical, point)


def _set_morphology_dependent_locations(stimulus, cell):
    """Here we deal with morphology dependent locations"""

    def _get_stim(stimulus, sec_id):
        new_stim = deepcopy(stimulus)
        stim_split = stimulus["name"].split(".")
        new_stim["type"] = "nrnseclistcomp"
        new_stim["name"] = f"{'.'.join(stim_split[:-1])}_{sec_id}.{stim_split[-1]}"
        new_stim["sec_index"] = sec_id
        return new_stim

    new_stims = []
    if stimulus["type"] == "somadistanceapic":
        new_stims = [deepcopy(stimulus)]
        new_stims[0]["sec_index"] = _get_apical_point(cell)
        new_stims[0]["sec_name"] = seclist_to_sec.get(
            stimulus["seclist_name"], stimulus["seclist_name"]
        )

    elif stimulus["type"] == "terminal_sections":
        # all terminal sections
        for sec_id, section in enumerate(getattr(cell.icell, stimulus["seclist_name"])):
            if len(section.subtree()) == 1:
                new_stims.append(_get_stim(stimulus, sec_id))

    elif stimulus["type"] == "all_sections":
        # all section of given type
        for sec_id, section in enumerate(getattr(cell.icell, stimulus["seclist_name"])):
            new_stims.append(_get_stim(stimulus, sec_id))

    else:
        new_stims = [deepcopy(stimulus)]

    if len(new_stims) == 0 and stimulus["type"] in [
        "somadistanceapic",
        "terminal_sections",
        "all_sections",
    ]:
        logger.warning("We could not add a location for %s", stimulus)
    return new_stims


class FitnessCalculatorConfiguration:

    """The goal of this class is to store the results of an efeature extraction (efeatures
    and protocols) or to contain the results of a previous extraction retrieved from an access
    point. This object is used for the creation of the fitness calculator.
    """

    def __init__(
        self,
        efeatures=None,
        protocols=None,
        name_rmp_protocol=None,
        name_rin_protocol=None,
        threshold_efeature_std=None,
        validation_protocols=None,
        ion_variables=None,
    ):
        """Init.

        The arguments efeatures and protocols are expected to be in the format ued for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            efeatures (list of dict): contains the description of the efeatures of the model
                in the format returned by the method as_dict of the EFeatureConfiguration class:
                [
                    {"efel_feature_name": str, "protocol_name": str, "recording_name": str,
                    "mean": float, "std": float, "efel_settings": dict}
                ]
            protocols (list of dict): contains the description of the protocols of the model
                in the format returned by the method as_dict of the ProtocolConfiguration class:
                [
                    {"name": str, "stimuli": list of dict, "recordings": list of dict,
                    "validation": bool}
                ]
            name_rmp_protocol (str): name of protocol whose features are to be used as targets for
                the search of the RMP.
            name_rin_protocol (str): name of protocol whose features are to be used as targets for
                the search of the Rin.
            threshold_efeature_std (float): lower limit for the std expressed as a percentage of
                the mean of the features value (optional). Legacy.
            validation_protocols (list of str): name of the protocols used for validation only.
            ion_variables (list of str): ion current names and ionic concentration anmes
                for all available mechanisms
        """

        self.ion_variables = ion_variables

        if protocols is None:
            self.protocols = []
        else:
            self.protocols = [
                ProtocolConfiguration(**p, ion_variables=self.ion_variables) for p in protocols
            ]

        self.efeatures = []
        if efeatures is not None:
            for f in efeatures:
                f_dict = deepcopy(f)
                f_dict.pop("threshold_efeature_std")
                self.efeatures.append(
                    EFeatureConfiguration(
                        **f_dict,
                        threshold_efeature_std=f.get(
                            "threshold_efeature_std", threshold_efeature_std
                        ),
                    )
                )

        if validation_protocols is None:
            self.validation_protocols = []
        else:
            self.validation_protocols = validation_protocols

        self.name_rmp_protocol = name_rmp_protocol
        self.name_rin_protocol = name_rin_protocol

    def protocol_exist(self, protocol_name):
        return bool(p for p in self.protocols if p.name == protocol_name)

    def _add_bluepyefe_protocol(self, protocol_name, protocol):
        """"""

        # By default include somatic recording
        recordings = [
            {
                "type": "CompRecording",
                "name": f"{protocol_name}.soma.v",
                "location": "soma",
                "variable": "v",
            }
        ]

        stimulus = deepcopy(protocol["step"])
        stimulus["holding_current"] = protocol["holding"]["amp"]

        validation = protocol_name in self.validation_protocols

        tmp_protocol = ProtocolConfiguration(
            name=protocol_name,
            stimuli=[stimulus],
            recordings=recordings,
            validation=validation,
            ion_variables=self.ion_variables,
        )

        self.protocols.append(tmp_protocol)

    def _add_bluepyefe_efeature(self, feature, protocol_name, recording, threshold_efeature_std):
        """"""

        recording_name = "soma.v" if recording == "soma" else recording

        tmp_feature = EFeatureConfiguration(
            efel_feature_name=feature["feature"],
            protocol_name=protocol_name,
            recording_name=recording_name,
            mean=feature["val"][0],
            std=feature["val"][1],
            efeature_name=feature.get("efeature_name", None),
            efel_settings=feature.get("efel_settings", {}),
            threshold_efeature_std=threshold_efeature_std,
        )

        if protocol_name == self.name_rmp_protocol and feature["feature"] == "voltage_base":
            tmp_feature.protocol_name = "RMPProtocol"
            tmp_feature.efel_feature_name = "steady_state_voltage_stimend"
        if protocol_name == self.name_rin_protocol and feature["feature"] == "voltage_base":
            tmp_feature.protocol_name = "SearchHoldingCurrent"
            tmp_feature.efel_feature_name = "steady_state_voltage_stimend"
        if (
            protocol_name == self.name_rin_protocol
            and feature["feature"] == "ohmic_input_resistance_vb_ssse"
        ):
            tmp_feature.protocol_name = "RinProtocol"

        if protocol_name not in PRE_PROTOCOLS and not self.protocol_exist(protocol_name):
            raise Exception(
                f"Trying to register efeatures for protocol {protocol_name},"
                " but this protocol does not exist"
            )

        self.efeatures.append(tmp_feature)

    def init_from_bluepyefe(self, efeatures, protocols, currents, threshold_efeature_std):
        """Fill the configuration using the output of BluePyEfe"""

        if (
            self.name_rmp_protocol
            and self.name_rmp_protocol not in efeatures
            and self.name_rmp_protocol != "all"
        ):
            raise Exception(
                f"The stimulus {self.name_rmp_protocol} requested for RMP "
                "computation couldn't be extracted from the ephys data."
            )
        if self.name_rin_protocol and self.name_rin_protocol not in efeatures:
            raise Exception(
                f"The stimulus {self.name_rin_protocol} requested for Rin "
                "computation couldn't be extracted from the ephys data."
            )

        self.protocols = []
        self.efeatures = []

        for protocol_name, protocol in protocols.items():
            self._add_bluepyefe_protocol(protocol_name, protocol)

        for protocol_name in efeatures:
            for recording in efeatures[protocol_name]:
                for feature in efeatures[protocol_name][recording]:
                    self._add_bluepyefe_efeature(
                        feature, protocol_name, recording, threshold_efeature_std
                    )

        # Add the current related features
        if currents:
            self.efeatures.append(
                EFeatureConfiguration(
                    efel_feature_name="bpo_holding_current",
                    protocol_name="SearchHoldingCurrent",
                    recording_name="soma.v",
                    mean=currents["holding_current"][0],
                    std=currents["holding_current"][1],
                    threshold_efeature_std=threshold_efeature_std,
                )
            )

            self.efeatures.append(
                EFeatureConfiguration(
                    efel_feature_name="bpo_threshold_current",
                    protocol_name="SearchThresholdCurrent",
                    recording_name="soma.v",
                    mean=currents["threshold_current"][0],
                    std=currents["threshold_current"][1],
                    threshold_efeature_std=threshold_efeature_std,
                )
            )

        self.remove_featureless_protocols()

    def _add_legacy_protocol(self, protocol_name, protocol):
        """"""

        # By default include somatic recording
        recordings = [
            {
                "type": "CompRecording",
                "name": f"{protocol_name}.soma.v",
                "location": "soma",
                "variable": "v",
            }
        ]

        if "extra_recordings" in protocol:
            for protocol_def in protocol["extra_recordings"]:
                recordings.append(protocol_def)
                protocol_def[
                    "name"
                ] = f"{protocol_name}.{protocol_def['name']}.{protocol_def['var']}"

        stimulus = deepcopy(protocol["stimuli"]["step"])
        if "holding" in protocol["stimuli"]:
            stimulus["holding_current"] = protocol["stimuli"]["holding"]["amp"]
        else:
            stimulus["holding_current"] = None

        validation = protocol_name in self.validation_protocols

        protocol_type = "Protocol"
        if "type" in protocol and protocol["type"] == "StepThresholdProtocol":
            protocol_type = "ThresholdBasedProtocol"

        stochasticity = True
        if "stage" in protocol:
            if not isinstance(protocol["stage"], list) or len(protocol["stage"]) != 1:
                raise NotImplementedError(
                    "Expected a list of size 1 for 'stage'"
                    f" of json file for protocol {protocol_name}, got {protocol['stage']}"
                )
            if 1 in protocol["stage"]:
                stochasticity = False

        tmp_protocol = ProtocolConfiguration(
            name=protocol_name,
            stimuli=[stimulus],
            recordings=recordings,
            validation=validation,
            ion_variables=self.ion_variables,
            protocol_type=protocol_type,
            stochasticity=stochasticity,
        )

        self.protocols.append(tmp_protocol)

    def _add_legacy_efeature(self, feature, protocol_name, recording, threshold_efeature_std):
        """"""

        recording_name = "soma.v" if recording == "soma" else recording

        tmp_feature = EFeatureConfiguration(
            efel_feature_name=feature["feature"],
            protocol_name=protocol_name,
            recording_name=recording_name,
            mean=feature["val"][0],
            std=feature["val"][1],
            efeature_name=feature.get("efeature_name", None),
            efel_settings=feature.get("efel_settings", {}),
            threshold_efeature_std=threshold_efeature_std,
        )

        if protocol_name == "Rin":
            if feature["feature"] == "ohmic_input_resistance_vb_ssse":
                tmp_feature.protocol_name = "RinProtocol"
            elif feature["feature"] == "voltage_base":
                tmp_feature.protocol_name = "SearchHoldingCurrent"
                tmp_feature.efel_feature_name = "steady_state_voltage_stimend"
            else:
                return

        if protocol_name == "RMP":
            if feature["feature"] == "voltage_base":
                tmp_feature.protocol_name = "RMPProtocol"
                tmp_feature.efel_feature_name = "steady_state_voltage_stimend"
            elif feature["feature"] == "Spikecount":
                tmp_feature.protocol_name = "RMPProtocol"
                tmp_feature.efel_feature_name = "Spikecount"
            else:
                return

        if protocol_name == "RinHoldCurrent":
            tmp_feature.protocol_name = "SearchHoldingCurrent"
        if protocol_name == "Threshold":
            tmp_feature.protocol_name = "SearchThresholdCurrent"

        if protocol_name not in PRE_PROTOCOLS and not self.protocol_exist(protocol_name):
            raise Exception(
                f"Trying to register efeatures for protocol {protocol_name},"
                " but this protocol does not exist"
            )

        self.efeatures.append(tmp_feature)

    def init_from_legacy_dict(self, efeatures, protocols, threshold_efeature_std):

        self.protocols = []
        self.efeatures = []

        if (
            self.name_rmp_protocol
            and self.name_rmp_protocol not in efeatures
            and "RMP" not in efeatures
        ):
            raise Exception(
                f"The protocol {self.name_rmp_protocol} requested for RMP nor RMPProtocol "
                "are present in your efeatures json file."
            )

        if (
            self.name_rin_protocol
            and self.name_rin_protocol not in efeatures
            and "Rin" not in efeatures
        ):
            raise Exception(
                f"The protocol {self.name_rin_protocol} requested for Rin nor RinProtocol "
                "are present in your efeatures json file."
            )

        for protocol_name, protocol in protocols.items():

            if protocol_name in PRE_PROTOCOLS + LEGACY_PRE_PROTOCOLS:
                continue

            self._add_legacy_protocol(protocol_name, protocol)
            validation = protocol.get("validation", False)

            if validation != self.protocols[-1].validation:
                raise Exception(
                    "The protocol was set as a validation protocol in the json but is not present "
                    "as a validation protocol in the settings"
                )

        for protocol_name in efeatures:
            for recording in efeatures[protocol_name]:
                for feature in efeatures[protocol_name][recording]:
                    self._add_legacy_efeature(
                        feature, protocol_name, recording, threshold_efeature_std
                    )

        self.remove_featureless_protocols()

    def remove_featureless_protocols(self):
        """Remove the protocols that o not have any matching efeatures"""

        to_remove = []

        for i, protocol in enumerate(self.protocols):

            for efeature in self.efeatures:
                if efeature.protocol_name == protocol.name:
                    break
            else:
                to_remove.append(i)

        self.protocols = [p for i, p in enumerate(self.protocols) if i not in to_remove]

    def configure_morphology_dependent_locations(self, _cell, simulator):
        """"""

        cell = deepcopy(_cell)
        cell.params = None
        cell.mechanisms = None
        cell.instantiate(sim=simulator)

        # TODO: THE SAME FOR STIMULI

        for i, protocol in enumerate(self.protocols):
            recordings = []
            for j, rec in enumerate(protocol.recordings):
                if rec["type"] != "CompRecording":
                    for _rec in _set_morphology_dependent_locations(rec, cell):
                        recordings.append(_rec)
                else:
                    recordings.append(self.protocols[i].recordings[j])
            self.protocols[i].recordings = recordings

        # if the loc of the recording is of the form axon*.v, we replace * by
        # all the corresponding int from the created recordings
        to_remove = []
        efeatures = []
        for i, efeature in enumerate(self.efeatures):
            loc_name, rec_name = efeature.recording_name.split(".")
            if loc_name[-1] == "*":
                to_remove.append(i)
                protocol = next(p for p in self.protocols if p.name == efeature.protocol_name)
                for rec in protocol.recordings:
                    base_rec_name = rec["name"].split(".")[1]
                    if base_rec_name.startswith(loc_name[:-1]):
                        efeatures.append(deepcopy(efeature))
                        efeatures[-1].recording_name = f"{base_rec_name}.{rec_name}"

        self.efeatures = [f for i, f in enumerate(self.efeatures) if i not in to_remove] + efeatures

    def as_dict(self):
        """Used for the storage of the configuration"""

        return {
            "efeatures": [e.as_dict() for e in self.efeatures],
            "protocols": [p.as_dict() for p in self.protocols],
        }

    def __str__(self):
        """String representation"""

        str_form = "Fitness Calculator Configuration:\n\n"

        str_form += "Protocols:\n"
        for p in self.protocols:
            str_form += f"   {p.as_dict()}\n"

        str_form += "EFeatures:\n"
        for f in self.efeatures:
            str_form += f"   {f.as_dict()}\n"

        return str_form
