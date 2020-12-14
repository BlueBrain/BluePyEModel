"""Evaluator module."""
import logging
from collections import defaultdict

import numpy

from bluepyopt.ephys.recordings import CompRecording
from bluepyopt.ephys.simulators import NrnSimulator
from bluepyopt.ephys.evaluators import CellEvaluator
from bluepyopt.ephys.objectivescalculators import ObjectivesCalculator
from bluepyopt.ephys.locations import (
    NrnSeclistCompLocation,
    NrnSomaDistanceCompLocation,
    NrnSecSomaDistanceCompLocation,
)
from bluepyopt.evaluators import Evaluator
from bluepyopt.ephys.efeatures import eFELFeature
from bluepyopt.ephys.objectives import SingletonObjective

from .protocols import (
    MainProtocol,
    RMPProtocol,
    RinProtocol,
    SearchHoldingCurrent,
    SearchThresholdCurrent,
    StepProtocol,
    StepThresholdProtocol,
)
from ..ecode import eCodes


logger = logging.getLogger(__name__)

soma_loc = NrnSeclistCompLocation(name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5)
# this location can be used to record at ais, using the option ais_recording below
ais_loc = NrnSeclistCompLocation(name="soma", seclist_name="axonal", sec_index=0, comp_x=0.5)


class MultiEvaluator(Evaluator):

    """Multiple cell evaluator"""

    def __init__(
        self,
        evaluators=None,
        sim=None,
    ):
        """Constructor

        Args:
            evaluators (list): list of CellModel evaluators
            sim (NrnSimulator): simulator (usually NrnSimulator object)
        """

        self.sim = sim
        self.evaluators = evaluators
        # these are identical for all models. Better solution available?
        self.param_names = self.evaluators[0].param_names

        # loop objectives for all evaluators, rename based on evaluators
        objectives = [
            objective for evaluator in self.evaluators for objective in evaluator.objectives
        ]
        params = self.evaluators[0].cell_model.params_by_names(self.param_names)
        super().__init__(objectives, params)

    def param_dict(self, param_array):
        """Convert param_array in param_dict"""
        return dict(zip(self.param_names, param_array))

    def objective_dict(self, objective_array):
        """Convert objective_array in objective_dict"""
        objective_names = [objective.name for objective in self.objectives]

        if len(objective_names) != len(objective_array):
            raise Exception(
                "MultiEvaluator: list given to objective_dict() " "has wrong number of objectives"
            )

        return dict(zip(objective_names, objective_array))

    def objective_list(self, objective_dict):
        """Convert objective_dict in objective_list"""
        return [objective_dict[objective.name] for objective in self.objectives]

    def evaluate_with_dicts(self, param_dict=None):
        """Run evaluation with dict as input and output"""
        scores = defaultdict(float)
        for evaluator in self.evaluators:
            score = evaluator.evaluate_with_dicts(param_dict=param_dict)
            for score_name in score:
                scores[score_name] += score[score_name]
        return scores

    def evaluate_with_lists(self, params=None):
        """Run evaluation with lists as input and outputs"""
        param_dict = self.param_dict(params)
        obj_dict = self.evaluate_with_dicts(param_dict=param_dict)
        return self.objective_list(obj_dict)

    def evaluate(self, param_list=None):
        """Run evaluation with lists as input and outputs"""
        return self.evaluate_with_lists(param_list)

    def __str__(self):
        content = "multi cell evaluator:\n"
        content += "  evaluators:\n"
        for evaluator in self.evaluators:
            content += "    %s\n" % str(evaluator)
        return content


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

    def calculate_features(self, responses, raise_warnings=False):
        """Calculate feature value"""

        if self.efel_feature_name.startswith("bpo_"):
            feature_values = numpy.array(self.calculate_bpo_feature(responses))

        else:
            efel_trace = self._construct_efel_trace(responses)

            if efel_trace is None:
                feature_values = None
            else:
                self._setup_efel()

                import efel

                values = efel.getFeatureValues(
                    [efel_trace],
                    [self.efel_feature_name],
                    raise_warnings=raise_warnings,
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

            feature_values = self.calculate_features(responses)
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

        return score


def define_feature(
    feature_definition,
    stim_start=None,
    stim_end=None,
    stim_amp=None,
    protocol_name=None,
    recording_name=None,
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

    Returns:
        eFELFeatureExtra
    """

    efel_feature_name = feature_definition["feature"]
    meanstd = feature_definition["val"]

    feature_name = "%s.%s.%s" % (
        protocol_name,
        recording_name,
        efel_feature_name,
    )

    if meanstd[1] < 0.01 * meanstd[0]:
        logger.warning(
            "E-feature %s has a standard deviation inferior to 1%% of its mean.", feature_name
        )

    if protocol_name:
        recording_names = {"": "%s.%s" % (protocol_name, recording_name)}
    else:
        recording_names = {"": recording_name}

    if "strict_stim" in feature_definition:
        strict_stim = feature_definition["strict_stim"]
    else:
        strict_stim = True

    if "threshold" in feature_definition:
        threshold = feature_definition["threshold"]
    else:
        threshold = -30

    feature = eFELFeatureBPEM(
        feature_name,
        efel_feature_name=efel_feature_name,
        recording_names=recording_names,
        stim_start=stim_start,
        stim_end=stim_end,
        exp_mean=meanstd[0],
        exp_std=meanstd[1],
        stimulus_current=stim_amp,
        threshold=threshold,
        int_settings={"strict_stiminterval": strict_stim},
    )

    return feature


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

    for k in eCodes:
        if k in name.lower():
            stimulus = eCodes[k](location=soma_loc, **protocol_definition["stimuli"])
            break
    else:
        raise KeyError(
            "There is no eCode linked to the stimulus name {}. "
            "See ecode/__init__.py for the available stimuli "
            "names".format(name.lower())
        )

    if protocol_definition["type"] == "StepThresholdProtocol":
        return StepThresholdProtocol(
            name=name,
            thresh_perc=protocol_definition["stimuli"]["thresh_perc"],
            stimulus=stimulus,
            recordings=recordings,
            stochasticity=stochasticity,
        )

    if protocol_definition["type"] == "StepProtocol":
        return StepProtocol(
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            stochasticity=stochasticity,
        )

    raise Exception('Protocol type "{}" unknown.'.format(protocol_definition["type"]))


def get_features_by_name(list_features, name):
    """Get a feature from its name/"""

    for feature in list_features:
        if feature["feature"] == name:
            return feature

    return None


def define_RMP_protocol(features_definition):
    """Define the resting membrane potential protocol"""

    feature_def = get_features_by_name(
        features_definition["RMPProtocol"]["soma.v"], "steady_state_voltage_stimend"
    )

    if feature_def:

        target_voltage = define_feature(
            feature_def,
            protocol_name="RMPProtocol",
            recording_name="soma.v",
        )

        protocol = RMPProtocol(name="RMPProtocol", location=soma_loc, target_voltage=target_voltage)

        return protocol, [target_voltage]

    logger.debug(
        "steady_state_voltage_stimend not present in the feature_definition"
        " dictionnary for RMPProtocol"
    )

    return None, []


def define_Rin_protocol(features_definition, ais_recording=False):
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
        )

        location = soma_loc if not ais_recording else ais_loc
        protocol = RinProtocol(name="RinProtocol", location=location, target_rin=target_rin)

        return protocol, [target_rin]

    logger.debug(
        "ohmic_input_resistance_vb_ssse not present in the feature_definition"
        " dictionnary for RinProtocol"
    )
    return None, []


def define_holding_protocol(features_definition):
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
        )
        target_holding_current = define_feature(
            def_holding_current,
            protocol_name="",
            recording_name="bpo_holding_current",
        )

        protocol = SearchHoldingCurrent(
            name="SearchHoldingCurrent",
            location=soma_loc,
            target_voltage=target_holding_voltage,
            target_holding=target_holding_current,
        )

        return protocol, [target_holding_current]

    logger.debug(
        "steady_state_voltage_stimend or bpo_holding_current not present"
        " in the feature_definition dictionnary for SearchHoldingCurrent"
    )
    return None, []


def define_threshold_protocol(features_definition):
    """Define the search threshold current protocol"""

    feature_def = get_features_by_name(
        features_definition["SearchThresholdCurrent"]["soma.v"], "bpo_threshold_current"
    )

    if feature_def:

        target_threshold = define_feature(
            feature_def,
            protocol_name="",
            recording_name="bpo_threshold_current",
        )

        protocol = SearchThresholdCurrent(
            name="SearchThresholdCurrent", location=soma_loc, target_threshold=target_threshold
        )

        return protocol, [target_threshold]

    logger.debug(
        "bpo_threshold_current not present in the feature_definition dictionnary"
        " for SearchThresholdCurrent"
    )
    return None, []


def define_main_protocol(  # pylint: disable=R0912,R0915,R0914,R1702
    protocols_definition,
    features_definition,
    stochasticity=True,
    ais_recording=False,
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
            f_definition = features_definition[name]
            for recording_name, feature_configs in f_definition.items():

                for f in feature_configs:

                    stim_amp = protocol.step_amplitude

                    if "stim_start" and "stim_end" in f:
                        stim_start = f["stim_start"]
                        stim_end = f["stim_end"]
                    else:
                        stim_start = protocol.stim_start
                        stim_end = protocol.stim_end

                        if "bAP" in name:
                            stim_end = protocol.total_duration

                    feature = define_feature(
                        f, stim_start, stim_end, stim_amp, protocol.name, recording_name
                    )
                    features.append(feature)

    rmp_protocol, rmp_features = define_RMP_protocol(features_definition)
    rin_protocol, rin_features = define_Rin_protocol(
        features_definition, ais_recording=ais_recording
    )
    search_holding_protocol, hold_features = define_holding_protocol(features_definition)
    search_threshold_protocol, thres_features = define_threshold_protocol(features_definition)

    features += thres_features + hold_features + rin_features + rmp_features

    main_protocol = MainProtocol(
        "Main",
        rmp_protocol,
        rin_protocol,
        search_holding_protocol,
        search_threshold_protocol,
        threshold_protocols=threshold_protocols,
        other_protocols=other_protocols,
        score_threshold=10.0,
    )

    return main_protocol, features


def define_fitness_calculator(features):
    """Creates the objectives calculator.

    Args:
        features (list): list of EFeature.

    Returns:
        ObjectivesCalculator
    """
    return ObjectivesCalculator([SingletonObjective(feat.name, feat) for feat in features])


def get_simulator(stochasticity, cell_models):
    """Get NrnSimulator

    Args:
        stochasticity (Bool): allow the use of simulator for stochastic channels
        cell_models (list): List of CellModel to detect if any stochastic channels are present
    """
    if stochasticity:
        for cell_model in cell_models:
            for mechanism in cell_model.mechanisms:
                if not mechanism.deterministic:
                    return NrnSimulator(dt=0.025, cvode_active=False)
    return NrnSimulator()


def create_evaluator(
    cell_model,
    protocols_definition,
    features_definition,
    simulator=None,
    stochasticity=True,
    timeout=None,
):
    """Creates an evaluator for a cell model/protocols/e-feature set

    Args:
        cell_model (CellModel): cell mode
        protocols_definition (dict): protocols and their definition
        features_definition (dict): features means and stds
        simulator (NrnSimulator): simulator, is None, one will be created
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic

    Returns:
        CellEvaluator
    """
    main_protocol, features = define_main_protocol(
        protocols_definition,
        features_definition,
        stochasticity,
    )

    fitness_calculator = define_fitness_calculator(features)
    fitness_protocols = {"main_protocol": main_protocol}

    param_names = [param.name for param in cell_model.params.values() if not param.frozen]

    if simulator is None:
        simulator = get_simulator(stochasticity, [cell_model])

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


def create_evaluators(
    cell_models, protocols_definition, features_definition, stochasticity=False, timeout=None
):
    """Create a multi-evaluator built from a list of evaluators (one for each
    cell model). The protocols and e-features will be the same for all the
    cell models.

    Args:
        cell_models (list): list of cell models
        protocols_definition (dict): protocols and their definition
        features_definition (dict): features means and stds
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic

    Returns:
        MultiEvaluator
    """
    simulator = get_simulator(stochasticity, cell_models)
    cell_evals = [
        create_evaluator(
            cell_model, protocols_definition, features_definition, simulator, stochasticity, timeout
        )
        for cell_model in cell_models
    ]

    return MultiEvaluator(evaluators=cell_evals, sim=simulator)
