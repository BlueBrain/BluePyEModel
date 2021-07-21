"""Evaluator module."""
import logging
import math
from copy import deepcopy

import numpy
from bluepyopt.ephys.efeatures import eFELFeature
from bluepyopt.ephys.evaluators import CellEvaluator
from bluepyopt.ephys.locations import NrnSeclistCompLocation
from bluepyopt.ephys.locations import NrnSecSomaDistanceCompLocation
from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation
from bluepyopt.ephys.objectives import SingletonObjective
from bluepyopt.ephys.objectivescalculators import ObjectivesCalculator
from bluepyopt.ephys.recordings import CompRecording
from bluepyopt.ephys.simulators import NrnSimulator

from ..ecode import eCodes
from .protocols import BPEM_Protocol
from .protocols import MainProtocol
from .protocols import RinProtocol
from .protocols import RMPProtocol
from .protocols import SearchHoldingCurrent
from .protocols import SearchThresholdCurrent

logger = logging.getLogger(__name__)

soma_loc = NrnSeclistCompLocation(name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5)
# this location can be used to record at ais, using the option ais_recording below
ais_loc = NrnSeclistCompLocation(name="soma", seclist_name="axonal", sec_index=0, comp_x=0.5)
seclist_to_sec = {
    "somatic": "soma",
    "apical": "apic",
    "axonal": "axon",
    "myelinated": "myelin",
}


class eFELFeatureBPEM(eFELFeature):

    """eFEL feature extra"""

    SERIALIZED_FIELDS = (
        "name",
        "efel_feature_name",
        "recording_names",
        "stim_start",
        "stim_end",
        "exp_mean",
        "exp_std",
        "threshold",
        "comment",
    )

    def __init__(
        self,
        name,
        efel_feature_name=None,
        recording_names=None,
        stim_start=None,
        stim_end=None,
        exp_mean=None,
        exp_std=None,
        threshold=None,
        stimulus_current=None,
        comment="",
        interp_step=None,
        double_settings=None,
        int_settings=None,
        efel_settings=None,
    ):
        """Constructor

        Args:
            name (str): name of the eFELFeature object
            efel_feature_name (str): name of the eFeature in the eFEL library
                (ex: 'AP1_peak')
            recording_names (dict): eFEL features can accept several recordings
                as input
            stim_start (float): stimulation start time (ms)
            stim_end (float): stimulation end time (ms)
            exp_mean (float): experimental mean of this eFeature
            exp_std(float): experimental standard deviation of this eFeature
            threshold(float): spike detection threshold (mV)
            comment (str): comment
        """

        super().__init__(
            name,
            efel_feature_name,
            recording_names,
            stim_start,
            stim_end,
            exp_mean,
            exp_std,
            threshold,
            stimulus_current,
            comment,
            interp_step,
            double_settings,
            int_settings,
            max_score=250.0,
        )
        self.efel_settings = efel_settings

    def calculate_bpo_feature(self, responses):
        """Return internal feature which is directly passed as a response"""

        if self.efel_feature_name not in responses:
            return None

        return responses[self.efel_feature_name]

    def calculate_bpo_score(self, responses):
        """Return score for bpo feature"""

        feature_value = self.calculate_bpo_feature(responses)

        if feature_value is None:
            return self.max_score

        return abs(feature_value - self.exp_mean) / self.exp_std

    def _construct_efel_trace(self, responses):
        """Construct trace that can be passed to eFEL"""

        trace = {}
        if "" not in self.recording_names:
            raise Exception("eFELFeature: '' needs to be in recording_names")
        for location_name, recording_name in self.recording_names.items():
            if location_name == "":
                postfix = ""
            else:
                postfix = ";%s" % location_name

            if recording_name not in responses:
                logger.debug(
                    "Recording named %s not found in responses %s", recording_name, str(responses)
                )
                return None

            if responses[self.recording_names[""]] is None or responses[recording_name] is None:
                return None
            trace["T%s" % postfix] = responses[self.recording_names[""]]["time"]
            trace["V%s" % postfix] = responses[recording_name]["voltage"]

            if callable(self.stim_start):
                trace["stim_start%s" % postfix] = [self.stim_start()]
            else:
                trace["stim_start%s" % postfix] = [self.stim_start]

            if callable(self.stim_end):
                trace["stim_end%s" % postfix] = [self.stim_end()]
            else:
                trace["stim_end%s" % postfix] = [self.stim_end]

        return trace

    def _setup_efel(self):
        """Set up efel before extracting the feature"""

        import efel

        efel.reset()

        if self.threshold is not None:
            efel.setThreshold(self.threshold)

        if self.stimulus_current is not None:
            if callable(self.stimulus_current):
                efel.setDoubleSetting("stimulus_current", self.stimulus_current())
            else:
                efel.setDoubleSetting("stimulus_current", self.stimulus_current)

        if self.interp_step is not None:
            efel.setDoubleSetting("interp_step", self.interp_step)

        if self.double_settings is not None:
            for setting_name, setting_value in self.double_settings.items():
                efel.setDoubleSetting(setting_name, setting_value)

        if self.int_settings is not None:
            for setting_name, setting_value in self.int_settings.items():
                efel.setIntSetting(setting_name, setting_value)

        if self.string_settings is not None:
            for setting_name, setting_value in self.string_settings.items():
                efel.setStrSetting(setting_name, setting_value)

    def calculate_feature(self, responses, raise_warnings=False):
        """Calculate feature value"""
        if self.efel_feature_name.startswith("bpo_"):
            feature_values = numpy.array([self.calculate_bpo_feature(responses)])

        else:
            efel_trace = self._construct_efel_trace(responses)

            if efel_trace is None:
                feature_values = None
            else:

                self._setup_efel()
                logger.debug("Amplitude for %s: %s", self.name, self.stimulus_current)
                import efel

                values = efel.getFeatureValues(
                    [efel_trace], [self.efel_feature_name], raise_warnings=raise_warnings
                )

                feature_values = values[0][self.efel_feature_name]

                efel.reset()

        logger.debug("Calculated values for %s: %s", self.name, str(feature_values))
        return feature_values

    def calculate_score(self, responses, trace_check=False):
        """Calculate the score"""

        if self.efel_feature_name.startswith("bpo_"):
            score = self.calculate_bpo_score(responses)

        elif self.exp_mean is None:
            score = 0

        else:

            feature_values = self.calculate_feature(responses)
            if (feature_values is None) or (len(feature_values) == 0):
                score = self.max_score
            else:
                score = (
                    numpy.sum(numpy.fabs(feature_values - self.exp_mean))
                    / self.exp_std
                    / len(feature_values)
                )
                logger.debug("Calculated score for %s: %f", self.name, score)

            score = numpy.min([score, self.max_score])

        if score is None or math.isnan(score):
            return self.max_score

        return score


def define_feature(
    feature_definition,
    stim_start=None,
    stim_end=None,
    stim_amp=None,
    protocol_name=None,
    recording_name=None,
    global_efel_settings=None,
    threshold_efeature_std=None,
):
    """Create a feature.

    Args:
        feature_definition (dict): see docstring of function
            define_main_protocol.
        stim_start (float): start of the time window on which this efeature
            will be computed.
        stim_end (float): end of the time window on which this efeature
            will be computed.
        stim_amp (float or None): current amplitude of the step.
        protocol_name (str): name of the protocol associated to this efeature.
        recording_name (str): name of the voltage recording (e.g.: "soma.v")
        global_efel_settings (dict): efel settings in the form
            {setting_name: setting_value}. If settings are also informed
            in the targets per efeature, the latter will have priority.
        threshold_efeature_std (float): if informed, compute the std as
            threshold_efeature_std * mean if std is < threshold_efeature_std * min.

    Returns:
        eFELFeatureExtra
    """

    if global_efel_settings is None:
        global_efel_settings = {}

    efel_feature_name = feature_definition["feature"]
    meanstd = feature_definition["val"]

    feature_name = "%s.%s.%s" % (
        protocol_name,
        recording_name,
        efel_feature_name,
    )

    std = meanstd[1]
    if threshold_efeature_std and std < abs(threshold_efeature_std * meanstd[0]):
        std = abs(threshold_efeature_std * meanstd[0])

    if protocol_name:
        recording_names = {"": "%s.%s" % (protocol_name, recording_name)}
    else:
        recording_names = {"": recording_name}

    efel_settings = {**global_efel_settings, **feature_definition.get("efel_settings", {})}
    efel_settings["stimulus_current"] = stim_amp

    if "strict_stim" in feature_definition:
        efel_settings["strict_stiminterval"] = feature_definition["strict_stim"]

    return eFELFeatureBPEM(
        feature_name,
        efel_feature_name=efel_feature_name,
        recording_names=recording_names,
        stim_start=stim_start,
        stim_end=stim_end,
        exp_mean=meanstd[0],
        exp_std=std,
        stimulus_current=stim_amp,
        efel_settings=efel_settings,
    )


def define_protocol(
    name,
    protocol_definition,
    stochasticity=True,
):
    """Create the protocol.

    Args:
        name (str): name of the protocol
        protocol_definition (dict): see docstring of function
            define_main_protocol
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic

    Returns:
        Protocol
    """
    # By default include somatic recording
    somav_recording = CompRecording(name="%s.soma.v" % name, location=soma_loc, variable="v")
    recordings = [somav_recording]

    if "extra_recordings" in protocol_definition:

        for recording_definition in protocol_definition["extra_recordings"]:

            if recording_definition["type"] == "somadistance":
                location = NrnSomaDistanceCompLocation(
                    name=recording_definition["name"],
                    soma_distance=recording_definition["somadistance"],
                    seclist_name=recording_definition["seclist_name"],
                )

            elif recording_definition["type"] == "somadistanceapic":
                location = NrnSecSomaDistanceCompLocation(
                    name=recording_definition["name"],
                    soma_distance=recording_definition["somadistance"],
                    sec_index=recording_definition["sec_index"],
                    sec_name=recording_definition["sec_name"],
                )

            elif recording_definition["type"] == "nrnseclistcomp":
                location = NrnSeclistCompLocation(
                    name=recording_definition["name"],
                    comp_x=recording_definition["comp_x"],
                    sec_index=recording_definition["sec_index"],
                    seclist_name=recording_definition["seclist_name"],
                )

            else:
                raise Exception("Recording type %s not supported" % recording_definition["type"])

            var = recording_definition["var"]
            recording = CompRecording(
                name="%s.%s.%s" % (name, location.name, var),
                location=location,
                variable=recording_definition["var"],
            )

            recordings.append(recording)

    for k, ecode in eCodes.items():
        if k in name.lower():

            if "stimuli" in protocol_definition:
                stim_def = protocol_definition["stimuli"]
            else:
                stim_def = {}

            stimulus = ecode(location=soma_loc, **stim_def)

            break

    else:
        raise KeyError(
            "There is no eCode linked to the stimulus name {}. "
            "See ecode/__init__.py for the available stimuli "
            "names".format(name.lower())
        )

    if protocol_definition["type"] == "StepThresholdProtocol":
        return BPEM_Protocol(
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            stochasticity=stochasticity,
            threshold_based=True,
        )

    if protocol_definition["type"] == "StepProtocol":
        return BPEM_Protocol(
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            stochasticity=stochasticity,
            threshold_based=False,
        )

    raise Exception('Protocol type "{}" unknown.'.format(protocol_definition["type"]))


def get_features_by_name(list_features, name):
    """Get a feature from its name"""

    for feature in list_features:
        if feature["feature"] == name:
            return feature

    return None


def define_RMP_protocol(features_definition, efel_settings, threshold_efeature_std):
    """Define the resting membrane potential protocol"""

    feature_def = get_features_by_name(
        features_definition["RMPProtocol"]["soma.v"], "steady_state_voltage_stimend"
    )

    if feature_def:

        target_voltage = define_feature(
            feature_def,
            protocol_name="RMPProtocol",
            recording_name="soma.v",
            global_efel_settings=efel_settings,
            threshold_efeature_std=threshold_efeature_std,
        )

        protocol = RMPProtocol(name="RMPProtocol", location=soma_loc, target_voltage=target_voltage)

        return protocol, [target_voltage]

    logger.warning(
        "steady_state_voltage_stimend not present in the feature_definition"
        " dictionnary for RMPProtocol"
    )

    return None, []


def define_Rin_protocol(
    features_definition, ais_recording=False, efel_settings=None, threshold_efeature_std=None
):
    """Define the Rin protocol.

    With ais_rcording=True, the recording will be at the first axonal section.
    """

    feature_def = get_features_by_name(
        features_definition["RinProtocol"]["soma.v"], "ohmic_input_resistance_vb_ssse"
    )

    if feature_def:

        target_rin = define_feature(
            feature_def,
            protocol_name="RinProtocol",
            recording_name="soma.v",
            global_efel_settings=efel_settings,
            threshold_efeature_std=threshold_efeature_std,
        )

        location = soma_loc if not ais_recording else ais_loc
        protocol = RinProtocol(name="RinProtocol", location=location, target_rin=target_rin)

        return protocol, [target_rin]

    logger.warning(
        "ohmic_input_resistance_vb_ssse not present in the feature_definition"
        " dictionnary for RinProtocol"
    )
    return None, []


def define_holding_protocol(features_definition, efel_settings=None, threshold_efeature_std=None):
    """Define the search holdinf current protocol"""
    def_holding_voltage = get_features_by_name(
        features_definition["SearchHoldingCurrent"]["soma.v"], "steady_state_voltage_stimend"
    )
    def_holding_current = get_features_by_name(
        features_definition["SearchHoldingCurrent"]["soma.v"], "bpo_holding_current"
    )

    if def_holding_voltage and def_holding_current:

        target_holding_voltage = define_feature(
            def_holding_voltage,
            protocol_name="SearchHoldingCurrent",
            recording_name="soma.v",
            global_efel_settings=efel_settings,
            threshold_efeature_std=threshold_efeature_std,
        )
        target_holding_current = define_feature(
            def_holding_current,
            protocol_name="",
            recording_name="bpo_holding_current",
            global_efel_settings=efel_settings,
            threshold_efeature_std=threshold_efeature_std,
        )

        protocol = SearchHoldingCurrent(
            name="SearchHoldingCurrent",
            location=soma_loc,
            target_voltage=target_holding_voltage,
            target_holding=target_holding_current,
        )

        return protocol, [target_holding_current]

    logger.warning(
        "steady_state_voltage_stimend or bpo_holding_current not present"
        " in the feature_definition dictionnary for SearchHoldingCurrent"
    )
    return None, []


def define_threshold_protocol(features_definition, efel_settings=None, threshold_efeature_std=None):
    """Define the search threshold current protocol"""

    feature_def = get_features_by_name(
        features_definition["SearchThresholdCurrent"]["soma.v"], "bpo_threshold_current"
    )

    if feature_def:

        target_threshold = define_feature(
            feature_def,
            protocol_name="",
            recording_name="bpo_threshold_current",
            global_efel_settings=efel_settings,
            threshold_efeature_std=threshold_efeature_std,
        )

        protocol = SearchThresholdCurrent(
            name="SearchThresholdCurrent", location=soma_loc, target_threshold=target_threshold
        )

        return protocol, [target_threshold]

    logger.warning(
        "bpo_threshold_current not present in the feature_definition dictionnary"
        " for SearchThresholdCurrent"
    )
    return None, []


def define_main_protocol(  # pylint: disable=R0912,R0915,R0914,R1702
    protocols_definition,
    features_definition,
    stochasticity=True,
    ais_recording=False,
    efel_settings=None,
    threshold_efeature_std=None,
    score_threshold=12.0,
):
    """Create the MainProtocol and the list of efeatures to use as objectives.

    The amplitude of the "threshold_protocols" depend on the computation of
    the current threshold while the "other_protocols" do not.

    Args:
        protocols_definition (dict): in the following format. The "type" field
            of a protocol can be StepProtocol, StepThresholdProtocol, RMP,
            RinHoldCurrent. If protocols with type StepThresholdProtocol are
            present, RMP and RinHoldCurrent should also be present.
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        ais_rcording (bool): if True all the soma recording will be at the first axonal section.
        efel_settings (dict): eFEl settings.
        threshold_efeature_std (float): if informed, compute the std as
            threshold_efeature_std * mean if std is < threshold_efeature_std * min.

    Returns:
    """
    threshold_protocols = {}
    other_protocols = {}
    features = []

    for name, definition in protocols_definition.items():
        
        protocol = define_protocol(name, definition, stochasticity)

        if definition["type"] == "StepThresholdProtocol":
            threshold_protocols[name] = protocol
        elif definition["type"] == "StepProtocol":
            other_protocols[name] = protocol
        else:
            raise Exception('Protocol type "{}" unknown.'.format(definition["type"]))

        # Define the efeatures associated to the protocol
        if name in features_definition:
            for recording_name, feature_configs in features_definition[name].items():

                for feature_config in feature_configs:

                    stim_amp = protocol.amplitude

                    stim_start = feature_config.get("stim_start", protocol.stim_start)
                    stim_end = feature_config.get("stim_end", protocol.stim_end)

                    if "bAP" in name:
                        stim_end = protocol.total_duration

                    feature = define_feature(
                        feature_config,
                        stim_start,
                        stim_end,
                        stim_amp,
                        protocol.name,
                        recording_name,
                        global_efel_settings=efel_settings,
                        threshold_efeature_std=threshold_efeature_std,
                    )
                    features.append(feature)

    rmp_protocol, rmp_features = define_RMP_protocol(
        features_definition, efel_settings, threshold_efeature_std
    )
    rin_protocol, rin_features = define_Rin_protocol(
        features_definition,
        ais_recording=ais_recording,
        efel_settings=efel_settings,
        threshold_efeature_std=threshold_efeature_std,
    )

    search_holding_protocol, hold_features = define_holding_protocol(
        features_definition, efel_settings, threshold_efeature_std
    )
    search_threshold_protocol, thres_features = define_threshold_protocol(
        features_definition, efel_settings, threshold_efeature_std
    )

    features += thres_features + hold_features + rin_features + rmp_features

    if threshold_protocols:

        for pre_protocol in [
            rmp_protocol,
            rin_protocol,
            search_holding_protocol,
            search_threshold_protocol,
        ]:
            if not pre_protocol:
                raise Exception(
                    "MainProtocol creation failed as there are "
                    "StepThresholdProtocols but a pre-protocol is"
                    " {}".format(pre_protocol)
                )

    main_protocol = MainProtocol(
        "Main",
        rmp_protocol,
        rin_protocol,
        search_holding_protocol,
        search_threshold_protocol,
        threshold_protocols=threshold_protocols,
        other_protocols=other_protocols,
        score_threshold=score_threshold,
    )

    return main_protocol, features


def define_fitness_calculator(features):
    """Creates the objectives calculator.

    Args:
        features (list): list of EFeature.

    Returns:
        ObjectivesCalculator
    """

    objectives = [SingletonObjective(feat.name, feat) for feat in features]

    return ObjectivesCalculator(objectives)


def get_simulator(stochasticity, cell_model):
    """Get NrnSimulator

    Args:
        stochasticity (Bool): allow the use of simulator for stochastic channels
        cell_model (CellModel): used to check if any stochastic channels are present
    """
    if stochasticity:
        for mechanism in cell_model.mechanisms:
            if not mechanism.deterministic:
                return NrnSimulator(dt=0.025, cvode_active=False)

        logger.warning(
            "Stochasticity is True but no mechanisms are stochastic. Switching to "
            "non-stochastic."
        )

    return NrnSimulator()


def _get_apical_point(cell):
    """Get the apical point isec usign automatic apical point detection."""
    from morph_tool import apical_point
    from morph_tool import nrnhines
    from morphio import Morphology

    point = apical_point.apical_point_position(Morphology(cell.morphology.morphology_path))
    return nrnhines.point_to_section_end(cell.icell.apical, point)


# pylint: disable=too-many-nested-blocks
def _handle_extra_recordings(protocols, features, _cell):
    """Here we deal with special types of recordings."""
    cell = deepcopy(_cell)
    cell.params = None
    cell.mechanisms = None
    cell.instantiate(sim=NrnSimulator())

    for protocol_name, protocol in protocols.items():
        if "extra_recordings" in protocol:
            extra_recordings = []
            for extra in protocol["extra_recordings"]:

                if extra["type"] == "somadistanceapic":
                    extra["sec_index"] = _get_apical_point(cell)
                    extra["sec_name"] = seclist_to_sec.get(
                        extra["seclist_name"], extra["seclist_name"]
                    )
                    extra_recordings.append(extra)

                elif extra["type"] == "terminal_sections":
                    # this recording is for highfreq protocol, to record on all terminal sections.
                    for sec_id, section in enumerate(getattr(cell.icell, extra["seclist_name"])):
                        if len(section.subtree()) == 1:
                            _extra = deepcopy(extra)
                            _extra["type"] = "nrnseclistcomp"
                            _extra["name"] = f"{extra['name']}{sec_id}"
                            _extra["sec_index"] = sec_id
                            extra_recordings.append(_extra)

                elif extra["type"] == "all_sections":
                    # this recording type records from all section of given type
                    for sec_id, section in enumerate(getattr(cell.icell, extra["seclist_name"])):
                        _extra = deepcopy(extra)
                        _extra["type"] = "nrnseclistcomp"
                        _extra["name"] = f"{extra['name']}{sec_id}"
                        _extra["sec_index"] = sec_id
                        extra_recordings.append(_extra)
                else:
                    extra_recordings.append(extra)

                protocols[protocol_name]["extra_recordings"] = extra_recordings

    features_out = deepcopy(features)
    for prot_name in features:
        for loc in features[prot_name]:
            for feat in features[prot_name][loc]:
                # if the loc of the recording is of the form axon*.v, we replace * by
                # all the corresponding int from the created protocols using their names
                _loc_name, _rec_name = loc.split(".")
                if _loc_name[-1] == "*":
                    features_out[prot_name].pop(loc, None)
                    for rec in protocols[prot_name].get("extra_recordings", []):
                        if rec["name"].startswith(_loc_name[:-1]):
                            _loc = rec["name"] + "." + _rec_name
                            features_out[prot_name][_loc].append(deepcopy(feat))
    return protocols, features_out


def create_evaluator(
    cell_model,
    protocols_definition,
    features_definition,
    stochasticity=True,
    timeout=None,
    efel_settings=None,
    threshold_efeature_std=None,
    score_threshold=12.0,
):
    """Creates an evaluator for a cell model/protocols/e-feature set

    Args:
        cell_model (CellModel): cell model
        protocols_definition (dict): protocols and their definition
        features_definition (dict): features means and stds
        stochasticity (bool): should the stochastic channels be stochastic or
            deterministic
        timeout (float): maximum time in second during which a protocol is
            allowed to run
        efel_settings (dict): efel settings in the form
            {setting_name: setting_value}. If settings are also informed
            in the targets per efeature, the latter will have priority.
        threshold_efeature_std (float): if informed, compute the std as
            threshold_efeature_std * mean if std is < threshold_efeature_std * min.
        score_threshold (float): threshold for score of protocols to stop evaluations

    Returns:
        CellEvaluator
    """

    protocols_definition, features_definition = _handle_extra_recordings(
        protocols_definition, features_definition, cell_model
    )

    main_protocol, features = define_main_protocol(
        protocols_definition,
        features_definition,
        stochasticity,
        efel_settings=efel_settings,
        threshold_efeature_std=threshold_efeature_std,
        score_threshold=score_threshold,
    )

    fitness_calculator = define_fitness_calculator(features)
    fitness_protocols = {"main_protocol": main_protocol}

    param_names = [param.name for param in cell_model.params.values() if not param.frozen]

    simulator = get_simulator(stochasticity, cell_model)

    cell_eval = CellEvaluator(
        cell_model=cell_model,
        param_names=param_names,
        fitness_protocols=fitness_protocols,
        fitness_calculator=fitness_calculator,
        sim=simulator,
        use_params_for_seed=True,
        timeout=timeout,
    )
    cell_eval.prefix = cell_model.name

    return cell_eval
