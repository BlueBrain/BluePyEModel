"""FitnessCalculatorConfiguration"""

"""
Copyright 2023, EPFL/Blue Brain Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from copy import deepcopy

from bluepyopt.ephys.locations import EPhysLocInstantiateException

from bluepyemodel.evaluation.efeature_configuration import EFeatureConfiguration
from bluepyemodel.evaluation.evaluator import LEGACY_PRE_PROTOCOLS
from bluepyemodel.evaluation.evaluator import PRE_PROTOCOLS
from bluepyemodel.evaluation.evaluator import define_location
from bluepyemodel.evaluation.evaluator import seclist_to_sec
from bluepyemodel.evaluation.protocol_configuration import ProtocolConfiguration
from bluepyemodel.tools.utils import are_same_protocol
from bluepyemodel.tools.utils import get_mapped_protocol_name

logger = logging.getLogger(__name__)


def _set_morphology_dependent_locations(recording, cell):
    """Here we deal with morphology dependent locations"""

    def _get_rec(recording, sec_id):
        new_rec = deepcopy(recording)
        rec_split = recording["name"].split(".")
        new_rec["type"] = "nrnseclistcomp"
        new_rec["name"] = f"{'.'.join(rec_split[:-1])}_{sec_id}.{rec_split[-1]}"
        new_rec["sec_index"] = sec_id
        return new_rec

    new_recs = []
    if recording["type"] == "somadistanceapic":
        new_recs = [deepcopy(recording)]
        new_recs[0]["sec_name"] = seclist_to_sec.get(
            recording["seclist_name"], recording["seclist_name"]
        )

    elif recording["type"] == "terminal_sections":
        # all terminal sections
        for sec_id, section in enumerate(getattr(cell.icell, recording["seclist_name"])):
            if len(section.subtree()) == 1:
                new_recs.append(_get_rec(recording, sec_id))

    elif recording["type"] == "all_sections":
        # all section of given type
        for sec_id, section in enumerate(getattr(cell.icell, recording["seclist_name"])):
            new_recs.append(_get_rec(recording, sec_id))

    else:
        new_recs = [deepcopy(recording)]

    if len(new_recs) == 0 and recording["type"] in [
        "somadistanceapic",
        "terminal_sections",
        "all_sections",
    ]:
        logger.warning("We could not add a location for %s", recording)
    return new_recs


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
        default_std_value=1e-3,
        validation_protocols=None,
        stochasticity=False,
        ion_variables=None,
    ):
        """Init.

        The arguments efeatures and protocols are expected to be in the format used for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            efeatures (list of dict): contains the description of the efeatures of the model
                in the format returned by the method as_dict of the EFeatureConfiguration class.
                Each unpacked dict will also act as the input to the EFeatureConfiguration class,
                details can be found in that class docstring. Below an example:

                .. code-block::

                    [
                        {
                            "efel_feature_name": "Spikecount",
                            "protocol_name": "IDrest_130",
                            "recording_name": "soma.v",
                            "mean": 6.026,
                            "original_std": 4.016,
                            "efeature_name": "Spikecount",
                            "weight": 1.0,  # optional
                            "efel_settings": {
                                "strict_stiminterval": true,
                                "Threshold": -30.0,
                                "interp_step": 0.025,
                                "stim_start": 700.0,
                                "stim_end": 2700.0
                            }
                        }
                    ]

            protocols (list of dict): contains the description of the protocols of the model
                in the format returned by the method as_dict of the ProtocolConfiguration class.
                Each unpacked dict will also act as the input to the ProtocolConfiguration class,
                details can be found in that class docstring. Below an example:

                .. code-block::

                    [
                        {
                            "name": "IDrest_130",
                            "stimuli": [
                                {
                                "delay": 663.3473451327434,
                                "amp": 0.3491827776582547,
                                "thresh_perc": 130.56103157894464,
                                "duration": 2053.050884955752,
                                "totduration": 3000.0,
                                "holding_current": -0.13307330411636328
                                }
                            ],
                            "recordings_from_config": [
                                {
                                "type": "CompRecording",
                                "name": "IDrest_130.soma.v",
                                "location": "soma",
                                "variable": "v"
                                }
                            ],
                            "validation": false,
                            "protocol_type": "ThresholdBasedProtocol",
                            "stochasticity": false
                        }
                    ]

            name_rmp_protocol (str or list): name and amplitude of protocol
                whose features are to be used as targets for the search of the RMP.
                e.g: ``["IV", 0]`` or ``"IV_0"``
            name_rin_protocol (str or list): name and amplitude of protocol
                whose features are to be used as targets for the search of the Rin.
                e.g: ``["IV", -20]`` or ``"IV_-20"``
            threshold_efeature_std (float): lower limit for the std expressed as a percentage of
                the mean of the features value (optional). Legacy.
             default_std_value (float): during and after extraction, this value will be used
                to replace the standard deviation if the standard deviation is 0.
            validation_protocols (list of str): name of the protocols used for validation only.
            stochasticity (bool or list of str): should channels behave stochastically if they can.
                If a list of protocol names is provided, the runs will be stochastic
                for these protocols, and deterministic for the other ones.
            ion_variables (list of str): ion current names and ionic concentration anmes
                for all available mechanisms
        """

        self.rmp_duration = 500.0
        self.rin_step_delay = 500.0
        self.rin_step_duration = 500.0
        self.rin_step_amp = -0.02
        self.rin_totduration = 1000.0
        self.search_holding_duration = 500.0
        self.search_threshold_step_delay = 500.0
        self.search_threshold_step_duration = 2000.0
        self.search_threshold_totduration = 3000.0
        self.ion_variables = ion_variables

        if protocols is None:
            self.protocols = []
        else:
            self.protocols = self.initialise_protocols(protocols)

        self.efeatures = []
        if efeatures is not None:
            self.efeatures = self.initialise_efeatures(
                efeatures, threshold_efeature_std, default_std_value
            )

        if validation_protocols is None:
            self.validation_protocols = []
        else:
            self.validation_protocols = validation_protocols

        self.stochasticity = stochasticity

        self.name_rmp_protocol = name_rmp_protocol
        self.name_rin_protocol = name_rin_protocol

        self.workflow_id = None
        self.default_std_value = default_std_value

    def initialise_protocols(self, protocols):
        """Initialise protocols from the FitnessCalculatorConfiguration format."""
        if protocols is None:
            return []
        return [ProtocolConfiguration(**p, ion_variables=self.ion_variables) for p in protocols]

    def initialise_efeatures(self, efeatures, threshold_efeature_std=None, default_std_value=1e-3):
        """Initialise efeatures from the FitnessCalculatorConfiguration format."""
        if efeatures is None:
            return []
        configured_efeatures = []
        for f in efeatures:
            f_dict = deepcopy(f)
            f_dict.pop("threshold_efeature_std", None)
            f_dict.pop("default_std_value", None)
            configured_efeatures.append(
                EFeatureConfiguration(
                    **f_dict,
                    threshold_efeature_std=f.get("threshold_efeature_std", threshold_efeature_std),
                    default_std_value=f.get("default_std_value", default_std_value),
                )
            )
        return configured_efeatures

    def protocol_exist(self, protocol_name):
        return bool(p for p in self.protocols if p.name == protocol_name)

    def check_stochasticity(self, protocol_name):
        """Check if stochasticity should be active for a given protocol"""
        if isinstance(self.stochasticity, list):
            return protocol_name in self.stochasticity
        return self.stochasticity

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

        validation = any(are_same_protocol(protocol_name, p) for p in self.validation_protocols)
        stochasticity = self.check_stochasticity(protocol_name)

        protocol_type = "Protocol"
        if self.name_rmp_protocol and self.name_rin_protocol:
            protocol_type = "ThresholdBasedProtocol"

        tmp_protocol = ProtocolConfiguration(
            name=protocol_name,
            stimuli=[stimulus],
            recordings_from_config=recordings,
            validation=validation,
            ion_variables=self.ion_variables,
            stochasticity=stochasticity,
            protocol_type=protocol_type,
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
            sample_size=feature.get("n", None),
            default_std_value=self.default_std_value,
        )

        if (
            are_same_protocol(self.name_rmp_protocol, protocol_name)
            and feature["feature"] == "voltage_base"
        ):
            tmp_feature.protocol_name = "RMPProtocol"
            tmp_feature.efel_feature_name = "steady_state_voltage_stimend"
        if (
            are_same_protocol(self.name_rin_protocol, protocol_name)
            and feature["feature"] == "voltage_base"
        ):
            tmp_feature.protocol_name = "SearchHoldingCurrent"
            tmp_feature.efel_feature_name = "steady_state_voltage_stimend"
        if (
            are_same_protocol(self.name_rin_protocol, protocol_name)
            and feature["feature"] == "ohmic_input_resistance_vb_ssse"
        ):
            tmp_feature.protocol_name = "RinProtocol"

        if protocol_name not in PRE_PROTOCOLS and not self.protocol_exist(protocol_name):
            raise ValueError(
                f"Trying to register efeatures for protocol {protocol_name},"
                " but this protocol does not exist"
            )

        self.efeatures.append(tmp_feature)

    def init_from_bluepyefe(
        self,
        efeatures,
        protocols,
        currents,
        threshold_efeature_std,
        protocols_mapping=None,
    ):
        """Fill the configuration using the output of BluePyEfe"""

        if self.name_rmp_protocol and not any(
            are_same_protocol(self.name_rmp_protocol, p) for p in efeatures
        ):
            raise ValueError(
                f"The stimulus {self.name_rmp_protocol} requested for RMP "
                "computation couldn't be extracted from the ephys data."
            )
        if self.name_rin_protocol and not any(
            are_same_protocol(self.name_rin_protocol, p) for p in efeatures
        ):
            raise ValueError(
                f"The stimulus {self.name_rin_protocol} requested for Rin "
                "computation couldn't be extracted from the ephys data."
            )

        self.protocols = []
        self.efeatures = []

        self.validation_protocols = [
            get_mapped_protocol_name(vp, protocols_mapping) for vp in self.validation_protocols
        ]

        for protocol_name, protocol in protocols.items():
            p_name = get_mapped_protocol_name(protocol_name, protocols_mapping)
            self._add_bluepyefe_protocol(p_name, protocol)

        for protocol_name in efeatures:
            for recording in efeatures[protocol_name]:
                for feature in efeatures[protocol_name][recording]:
                    p_name = get_mapped_protocol_name(protocol_name, protocols_mapping)
                    self._add_bluepyefe_efeature(feature, p_name, recording, threshold_efeature_std)

        # Add the current related features
        if currents and self.name_rmp_protocol and self.name_rin_protocol:
            self.efeatures.append(
                EFeatureConfiguration(
                    efel_feature_name="bpo_holding_current",
                    protocol_name="SearchHoldingCurrent",
                    recording_name="soma.v",
                    mean=currents["holding_current"][0],
                    std=currents["holding_current"][1],
                    threshold_efeature_std=threshold_efeature_std,
                    default_std_value=self.default_std_value,
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
                    default_std_value=self.default_std_value,
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
                protocol_def["name"] = (
                    f"{protocol_name}.{protocol_def['name']}.{protocol_def['var']}"
                )

        stimulus = deepcopy(protocol["stimuli"]["step"])
        if "holding" in protocol["stimuli"]:
            stimulus["holding_current"] = protocol["stimuli"]["holding"]["amp"]
        else:
            stimulus["holding_current"] = None

        validation = any(are_same_protocol(protocol_name, p) for p in self.validation_protocols)
        stochasticity = self.check_stochasticity(protocol_name)

        protocol_type = "Protocol"
        if "type" in protocol and protocol["type"] == "StepThresholdProtocol":
            protocol_type = "ThresholdBasedProtocol"

        tmp_protocol = ProtocolConfiguration(
            name=protocol_name,
            stimuli=[stimulus],
            recordings_from_config=recordings,
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
            default_std_value=self.default_std_value,
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
            raise ValueError(
                f"Trying to register efeatures for protocol {protocol_name},"
                " but this protocol does not exist"
            )

        self.efeatures.append(tmp_feature)

    def init_from_legacy_dict(self, efeatures, protocols, threshold_efeature_std):
        self.protocols = []
        self.efeatures = []

        if (
            self.name_rmp_protocol
            and not any(are_same_protocol(self.name_rmp_protocol, p) for p in efeatures)
            and "RMP" not in efeatures
        ):
            raise ValueError(
                f"The protocol {self.name_rmp_protocol} requested for RMP nor RMPProtocol "
                "are present in your efeatures json file."
            )

        if (
            self.name_rin_protocol
            and not any(are_same_protocol(self.name_rin_protocol, p) for p in efeatures)
            and "Rin" not in efeatures
        ):
            raise ValueError(
                f"The protocol {self.name_rin_protocol} requested for Rin nor RinProtocol "
                "are present in your efeatures json file."
            )

        for protocol_name, protocol in protocols.items():
            if protocol_name == "RMP":
                self.rmp_duration = protocol["stimuli"]["step"]["duration"]
            if protocol_name == "Rin":
                self.rin_step_delay = protocol["stimuli"]["step"]["delay"]
                self.rin_step_duration = protocol["stimuli"]["step"]["duration"]
                self.rin_step_amp = protocol["stimuli"]["step"]["amp"]
                self.rin_totduration = protocol["stimuli"]["step"]["totduration"]
            if protocol_name == "ThresholdDetection":
                self.search_threshold_step_delay = protocol["step_template"]["stimuli"]["step"][
                    "delay"
                ]
                self.search_threshold_step_duration = protocol["step_template"]["stimuli"]["step"][
                    "duration"
                ]
                self.search_threshold_totduration = protocol["step_template"]["stimuli"]["step"][
                    "totduration"
                ]

            if protocol_name in PRE_PROTOCOLS + LEGACY_PRE_PROTOCOLS:
                continue

            self._add_legacy_protocol(protocol_name, protocol)
            validation = protocol.get("validation", False)

            if validation != self.protocols[-1].validation:
                raise ValueError(
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
        """Remove the protocols that do not have any matching efeatures"""

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

        skipped_recordings = []
        for i, protocol in enumerate(self.protocols):
            recordings = []
            for j, rec in enumerate(protocol.recordings):
                if rec["type"] != "CompRecording":
                    for _rec in _set_morphology_dependent_locations(rec, cell):
                        try:
                            location = define_location(_rec)
                            location.instantiate(sim=simulator, icell=cell.icell)
                            recordings.append(_rec)
                        except EPhysLocInstantiateException:
                            logger.warning(
                                "Could not find %s, ignoring recording at this location",
                                location.name,
                            )
                            skipped_recordings.append(_rec["name"])
                else:
                    recordings.append(self.protocols[i].recordings[j])
            self.protocols[i].recordings = recordings

        to_remove = []
        efeatures = []
        for i, efeature in enumerate(self.efeatures):
            if isinstance(efeature.recording_name, str):
                # remove efeature associated to skipped recording
                for skiprec in skipped_recordings:
                    if f"{efeature.protocol_name}.{efeature.recording_name}" == skiprec:
                        to_remove.append(i)
                        logger.warning("Removing %s", efeature.name)
                        continue
                # if the loc of the recording is of the form axon*.v, we replace * by
                # all the corresponding int from the created recordings
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

    def get_related_nexus_ids(self):
        return {
            "generation": {
                "type": "Generation",
                "activity": {
                    "type": "Activity",
                    "followedWorkflow": {
                        "type": "EModelWorkflow",
                        "id": self.workflow_id,
                    },
                },
            }
        }

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
