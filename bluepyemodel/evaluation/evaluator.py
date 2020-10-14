import numpy
import logging
import bluepyopt
from collections import defaultdict

import bluepyopt.ephys as ephys
from bluepyopt.ephys.efeatures import eFELFeature
from bluepyopt.ephys.objectives import EFeatureObjective

from .protocols import (
    MainProtocol,
    StepProtocol,
    StepThresholdProtocol,
    SearchRinHoldingCurrent,
    SearchThresholdCurrent,
    RMPProtocol,
)

logger = logging.getLogger(__name__)

soma_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)


class NrnSomaDistanceCompLocation(ephys.locations.NrnSomaDistanceCompLocation):
    def __init__(
        self,
        name,
        soma_distance=None,
        seclist_name=None,
        comment="",
    ):

        super(NrnSomaDistanceCompLocation, self).__init__(
            name, soma_distance, seclist_name, comment
        )

    def instantiate(self, sim=None, icell=None):
        """Find the instantiate compartment"""

        soma = icell.soma[0]

        sim.neuron.h.distance(0, 0.5, sec=soma)

        iseclist = getattr(icell, self.seclist_name)

        icomp = None

        for isec in iseclist:
            start_distance = sim.neuron.h.distance(1, 0.0, sec=isec)
            end_distance = sim.neuron.h.distance(1, 1.0, sec=isec)

            min_distance = min(start_distance, end_distance)
            max_distance = max(start_distance, end_distance)

            if min_distance <= self.soma_distance <= end_distance:
                comp_x = float(self.soma_distance - min_distance) / (
                    max_distance - min_distance
                )

                icomp = isec(comp_x)
                seccomp = isec
                break

        if icomp is None:
            raise ephys.locations.EPhysLocInstantiateException(
                "No comp found at %s distance from soma" % self.soma_distance
            )

        logger.debug(
            "Using %s at distance %f, nseg %f, length %f"
            % (
                icomp,
                sim.neuron.h.distance(1, comp_x, sec=seccomp),
                seccomp.nseg,
                end_distance - start_distance,
            )
        )

        return icomp


class NrnSomaDistanceCompLocationApical(ephys.locations.NrnSomaDistanceCompLocation):
    def __init__(
        self,
        name,
        soma_distance=None,
        seclist_name=None,
        comment="",
        apical_point_isec=None,
    ):

        super(NrnSomaDistanceCompLocationApical, self).__init__(
            name, soma_distance, seclist_name, comment
        )
        self.apical_point_isec = apical_point_isec

    def instantiate(self, sim=None, icell=None):
        """Find the instantiate compartment"""
        if self.apical_point_isec is None:
            raise ephys.locations.EPhysLocInstantiateException(
                "No apical point was given"
            )

        apical_branch = []
        section = icell.apic[self.apical_point_isec]
        while True:
            name = str(section.name()).split(".")[-1]
            if "soma[0]" == name:
                break
            apical_branch.append(section)

            if sim.neuron.h.SectionRef(sec=section).has_parent():
                section = sim.neuron.h.SectionRef(sec=section).parent
            else:
                raise ephys.locations.EPhysLocInstantiateException(
                    "soma[0] was not reached from apical point"
                )

        soma = icell.soma[0]

        sim.neuron.h.distance(0, 0.5, sec=soma)

        icomp = None

        for isec in apical_branch:
            start_distance = sim.neuron.h.distance(1, 0.0, sec=isec)
            end_distance = sim.neuron.h.distance(1, 1.0, sec=isec)

            min_distance = min(start_distance, end_distance)
            max_distance = max(start_distance, end_distance)

            if min_distance <= self.soma_distance <= end_distance:
                comp_x = float(self.soma_distance - min_distance) / (
                    max_distance - min_distance
                )

                icomp = isec(comp_x)
                seccomp = isec

        if icomp is None:
            raise ephys.locations.EPhysLocInstantiateException(
                "No comp found at %s distance from soma" % self.soma_distance
            )

        logger.debug(
            "Using %s at distance %f"
            % (icomp, sim.neuron.h.distance(1, comp_x, sec=seccomp))
        )

        return icomp


class MultiEvaluator(bluepyopt.evaluators.Evaluator):

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
        objectives = []
        # loop objectives for all evaluators, rename based on evaluators
        for i, evaluator in enumerate(self.evaluators):
            for objective in evaluator.objectives:
                objectives.append(objective)

        # these are identical for all models. Better solution available?
        self.param_names = self.evaluators[0].param_names
        params = self.evaluators[0].cell_model.params_by_names(self.param_names)

        super(MultiEvaluator, self).__init__(objectives, params)

    def param_dict(self, param_array):
        """Convert param_array in param_dict"""
        param_dict = {}
        for param_name, param_value in zip(self.param_names, param_array):
            param_dict[param_name] = param_value

        return param_dict

    def objective_dict(self, objective_array):
        """Convert objective_array in objective_dict"""
        objective_dict = {}
        objective_names = [objective.name for objective in self.objectives]

        if len(objective_names) != len(objective_array):
            raise Exception(
                "MultiEvaluator: list given to objective_dict() "
                "has wrong number of objectives"
            )

        for objective_name, objective_value in zip(objective_names, objective_array):
            objective_dict[objective_name] = objective_value

        return objective_dict

    def objective_list(self, objective_dict):
        """Convert objective_dict in objective_list"""
        objective_list = []
        objective_names = [objective.name for objective in self.objectives]
        for objective_name in objective_names:
            objective_list.append(objective_dict[objective_name])
        return objective_list

    def evaluate_with_dicts(self, param_dict=None):
        """Run evaluation with dict as input and output"""
        scores = defaultdict(float)
        for evaluator in self.evaluators:
            score = evaluator.evaluate_with_dicts(param_dict=param_dict)
            for score_name in score:
                scores[score_name] += score[score_name]
        return scores

    def evaluate_with_lists(self, param_list=None):
        """Run evaluation with lists as input and outputs"""
        param_dict = self.param_dict(param_list)
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


class eFELFeatureExtra(eFELFeature):

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
        exp_vals=None,
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

        super(eFELFeatureExtra, self).__init__(
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
        )

        extra_features = [
            "spikerate_tau_jj_skip",
            "spikerate_drop_skip",
            "spikerate_tau_log_skip",
            "spikerate_tau_fit_skip",
        ]

        if self.efel_feature_name in extra_features:
            self.extra_feature_name = self.efel_feature_name
            self.efel_feature_name = "peak_time"
        else:
            self.extra_feature_name = None

        self.exp_vals = exp_vals

    def get_bpo_feature(self, responses):
        """Return internal feature which is directly passed as a response"""

        if self.efel_feature_name not in responses:
            return None
        else:
            return responses[self.efel_feature_name]

    def get_bpo_score(self, responses):
        """Return internal score which is directly passed as a response"""

        feature_value = self.get_bpo_feature(responses)
        if feature_value is None:
            score = 250.0
        else:
            score = abs(feature_value - self.exp_mean) / self.exp_std
        return score

    def calculate_features(self, responses, raise_warnings=False):
        """Calculate feature value"""

        if self.efel_feature_name.startswith("bpo_"):  # check if internal feature
            feature_values = numpy.array(self.get_bpo_feature(responses))
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

        # check if internal feature
        if self.efel_feature_name.startswith("bpo_"):
            score = self.get_bpo_score(responses)

        elif self.exp_mean is None:
            score = 0

        else:

            feature_values = self.calculate_features(responses)
            if (feature_values is None) or (len(feature_values) == 0):
                score = 250.0
            else:
                score = (
                    numpy.sum(numpy.fabs(feature_values - self.exp_mean))
                    / self.exp_std
                    / len(feature_values)
                )
                logger.debug("Calculated score for %s: %f", self.name, score)

            score = numpy.min([score, 250.0])
        return score


class SingletonWeightObjective(EFeatureObjective):

    """Single EPhys feature"""

    def __init__(self, name, feature, weight):
        """Constructor

        Args:
            name (str): name of this object
            feature (EFeature): single eFeature inside this objective
            weight ():
        """

        super(SingletonWeightObjective, self).__init__(name, [feature])
        self.weight = weight

    def calculate_score(self, responses):
        """Objective score"""

        return self.calculate_feature_scores(responses)[0] * self.weight

    def __str__(self):
        """String representation"""

        return "( %s ), weight:%f" % (self.features[0], self.weight)


def get_features_by_name(list_features, name):
    for feature in list_features:
        if feature["feature"] == name:
            return feature


def define_feature(
    feature_definition, stim_start, stim_end, stim_amp, protocol_name, recording_name
):
    """Create a feature.

    Args:
        feature_definition (dict): see docstring of function
            define_main_protocol_features.
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
    recording_names = {"": "%s.%s" % (protocol_name, recording_name)}

    if "strict_stim" in feature_definition:
        strict_stim = feature_definition["strict_stim"]
    else:
        strict_stim = True

    if "threshold" in feature_definition:
        threshold = feature_definition["threshold"]
    else:
        threshold = -30

    feature = eFELFeatureExtra(
        feature_name,
        efel_feature_name=efel_feature_name,
        recording_names=recording_names,
        stim_start=stim_start,
        stim_end=stim_end,
        exp_mean=meanstd[0],
        exp_std=meanstd[1],
        exp_vals=meanstd,
        stimulus_current=stim_amp,
        threshold=threshold,
        int_settings={"strict_stiminterval": strict_stim},
    )

    return feature


def define_protocol(
    name,
    protocol_definition,
    stochasticity=False,
    apical_point_isec=None,
):
    """Create the protocol.

    Args:
        name (str): name of the protocol
        protocol_definition (dict): see docstring of function
            define_main_protocol_features
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        apical_point_isec (isec): dendritic section at which the recordings
            takes place. Used only when "type" == "somadistanceapic".

    Returns:
        Protocol
    """

    # By default include somatic recording
    somav_recording = ephys.recordings.CompRecording(
        name="%s.soma.v" % name,
        location=soma_loc,
        variable="v",
    )
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
                location = NrnSomaDistanceCompLocationApical(
                    name=recording_definition["name"],
                    soma_distance=recording_definition["somadistance"],
                    seclist_name=recording_definition["seclist_name"],
                    apical_point_isec=apical_point_isec,
                )

            elif recording_definition["type"] == "nrnseclistcomp":
                location = ephys.locations.NrnSeclistCompLocation(
                    name=recording_definition["name"],
                    comp_x=recording_definition["comp_x"],
                    sec_index=recording_definition["sec_index"],
                    seclist_name=recording_definition["seclist_name"],
                )

            else:
                raise Exception(
                    "Recording type %s not supported" % recording_definition["type"]
                )

            var = recording_definition["var"]
            recording = ephys.recordings.CompRecording(
                name="%s.%s.%s" % (name, location.name, var),
                location=location,
                variable=recording_definition["var"],
            )
            recordings.append(recording)

    holding_def = protocol_definition["stimuli"]["holding"]
    holding_stimulus = ephys.stimuli.NrnSquarePulse(
        step_amplitude=holding_def["amp"],
        step_delay=holding_def["delay"],
        step_duration=holding_def["duration"],
        location=soma_loc,
        total_duration=holding_def["totduration"],
    )

    step_def = protocol_definition["stimuli"]["step"]
    step_stimulus = ephys.stimuli.NrnSquarePulse(
        step_amplitude=step_def["amp"],
        step_delay=step_def["delay"],
        step_duration=step_def["duration"],
        location=soma_loc,
        total_duration=step_def["totduration"],
    )

    if protocol_definition["type"] == "StepThresholdProtocol":
        return StepThresholdProtocol(
            name=name,
            thresh_perc=protocol_definition["stimuli"]["step"]["thresh_perc"],
            step_stimulus=step_stimulus,
            holding_stimulus=holding_stimulus,
            recordings=recordings,
            stochasticity=stochasticity,
        )

    elif protocol_definition["type"] == "StepProtocol":
        return StepProtocol(
            name=name,
            step_stimulus=step_stimulus,
            holding_stimulus=holding_stimulus,
            recordings=recordings,
            stochasticity=stochasticity,
        )

    elif protocol_definition["type"] == "RMP":
        protocol = RMPProtocol(
            name=name,
            step_stimulus=step_stimulus,
            holding_stimulus=holding_stimulus,
            recordings=recordings,
            stochasticity=stochasticity,
        )
        protocol.holding_stimulus.step_amplitude = 0.0
        return protocol

    elif protocol_definition["type"] == "RinHoldCurrent":
        protocol = StepProtocol(
            name=name,
            step_stimulus=step_stimulus,
            holding_stimulus=holding_stimulus,
            recordings=recordings,
            stochasticity=stochasticity,
        )
        tot_dur = (
            protocol.step_stimulus.step_delay
            + protocol.step_stimulus.step_duration
            + 100.0
        )
        protocol.holding_stimulus.total_duration = tot_dur
        protocol.step_stimulus.total_duration = tot_dur
        return SearchRinHoldingCurrent(
            name="SearchRinHoldingCurrent", protocol=protocol
        )

    else:
        raise Exception(
            'Protocol type "{}" unknown.'.format(protocol_definition["type"])
        )


def define_main_protocol_features(
    protocols_definition, features_definition, stochasticity=False
):
    """Create the MainProtocol and the list of efeatures to use as objectives.

    The amplitude of the "threshold_protocols" depend on the computation of
    the current threshold while the "other_protocols" do not.

    Args:
        protocols_definition (dict): in the following format. The "type" field
            of a protocol can be StepProtocol, StepThresholdProtocol, RMP,
            RinHoldCurrent. If protocols with type StepThresholdProtocol are
            present, RMP and RinHoldCurrent should also be present.
            protocols_definition = {
                "Rin": {
                    "type": "StepProtocol",
                    "stimuli": {
                        "step": {
                            "delay": 700,
                            "amp": -0.1097239395382074,
                            "thresh_perc": -41,
                            "duration": 1000,
                            "totduration": 3800
                        },
                        "holding": {
                            "delay": 0,
                            "amp": null,
                            "duration": 3100,
                            "totduration": 3800
                        }
                    }
                }
            }
        features_definition (dict): of the form
            features_definition =  {
                "APWaveform_200": {
                    "soma.v": [
                        {
                            "feature": "Spikecount",
                            "val": [1.64, 0.71],
                            "strict_stim":True
                        },
                        {
                            "feature": "AHP_depth",
                            "val": [19.67, 4.74],
                            "strict_stim": true
                        }
                }
            }
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic

    Returns:

    """

    rmp_protocol = None
    holding_rin_protocol = None
    search_threshold_protocol = None

    threshold_protocols = {}
    other_protocols = {}
    features = []

    for name, definition in protocols_definition.items():

        # Define holding for the protocols that do not have it
        if "holding" not in definition:
            dur = definition["stimuli"]["step"]["totduration"]
            holding = {"delay": 0, "amp": None, "duration": dur, "totduration": dur}
            definition["stimuli"]["holding"] = holding

        protocol = define_protocol(name, definition, stochasticity)

        if definition["type"] == "StepThresholdProtocol":
            threshold_protocols[name] = protocol
        elif definition["type"] == "StepProtocol":
            other_protocols[name] = protocol
        elif definition["type"] == "RMP":
            rmp_protocol = protocol
        elif definition["type"] == "RinHoldCurrent":
            holding_rin_protocol = protocol
        else:
            raise Exception('Protocol type "{}" unknown.'.format(definition["type"]))

        # Define the efeatures associated to the protocol
        f_definition = features_definition[name]
        for recording_name, feature_configs in f_definition.items():

            for f in feature_configs:

                if hasattr(protocol, "stim_start"):

                    stim_amp = protocol.step_amplitude
                    stim_start = protocol.stim_start
                    if "bAP" in name:
                        # bAP response can be after stimulus
                        stim_end = protocol.total_duration
                    else:
                        stim_end = protocol.stim_end

                else:
                    stim_amp = None
                    stim_start = None
                    stim_end = None

                feature = define_feature(
                    f, stim_start, stim_end, stim_amp, protocol.name, recording_name
                )
                features.append(feature)

                if definition["type"] == "RinHoldCurrent":
                    if feature.efel_feature_name == "voltage_base":
                        holding_rin_protocol.target_voltage = feature
                    elif feature.efel_feature_name == "ohmic_input_resistance_vb_ssse":
                        holding_rin_protocol.target_Rin = feature
                    elif feature.efel_feature_name == "bpo_holding_current":
                        holding_rin_protocol.target_current = feature

                elif definition["type"] == "RMP":
                    if feature.efel_feature_name == "voltage_base":
                        rmp_protocol.target_voltage = feature

    if holding_rin_protocol or rmp_protocol:
        assert rmp_protocol.target_voltage
        assert holding_rin_protocol and rmp_protocol
        assert holding_rin_protocol.target_voltage
        assert holding_rin_protocol.target_Rin

    # Define the search for the threshold current
    if holding_rin_protocol and rmp_protocol:

        feature_def = get_features_by_name(
            features_definition["Threshold"]["soma.v"], "bpo_threshold_current"
        )
        threshold = define_feature(
            feature_def,
            stim_start=0,
            stim_end=0,
            stim_amp=0,
            protocol_name="",
            recording_name="threshold_current",
        )
        features.append(threshold)

        # Define the protocol searching for the spiking current
        search_threshold_protocol = SearchThresholdCurrent(
            name="SearchThreshold", location=soma_loc, target_threshold=threshold
        )

    main_protocol = MainProtocol(
        name="Main",
        rmp_protocol=rmp_protocol,
        holding_rin_protocol=holding_rin_protocol,
        search_threshold_protocol=search_threshold_protocol,
        threshold_protocols=threshold_protocols,
        other_protocols=other_protocols,
        score_threshold=None,
    )

    return main_protocol, features


def define_fitness_calculator(features):
    """Creates the objectives calculator.

    Args:
        features (list): list of EFeature.

    Returns:
        ObjectivesCalculator
    """
    objectives = []
    for feat in features:
        objective = SingletonWeightObjective(feat.name, feat, weight=1)
        objectives.append(objective)
    return ephys.objectivescalculators.ObjectivesCalculator(objectives)


def get_simulator(stochasticity):
    """Get NrnSimulator."""
    if stochasticity:
        return ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
    return ephys.simulators.NrnSimulator()


def create_evaluator(
    cell_model,
    protocols_definition,
    features_definition,
    simulator=None,
    stochasticity=False,
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
    main_protocol, features = define_main_protocol_features(
        protocols_definition,
        features_definition,
        stochasticity,
    )

    fitness_calculator = define_fitness_calculator(features)
    fitness_protocols = {"main_protocol": main_protocol}

    param_names = [
        param.name for param in cell_model.params.values() if not param.frozen
    ]

    if simulator is None:
        simulator = get_simulator(stochasticity)

    cell_eval = ephys.evaluators.CellEvaluator(
        cell_model=cell_model,
        param_names=param_names,
        fitness_protocols=fitness_protocols,
        fitness_calculator=fitness_calculator,
        sim=simulator,
        use_params_for_seed=True,
    )
    cell_eval.prefix = cell_model.name

    return cell_eval


def create_evaluators(
    cell_models, protocols_definition, features_definition, stochasticity=False
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
    simulator = get_simulator(stochasticity)
    cell_evals = [
        create_evaluator(
            cell_model,
            protocols_definition,
            features_definition,
            simulator,
            stochasticity,
        )
        for cell_model in cell_models
    ]

    return MultiEvaluator(evaluators=cell_evals, sim=simulator)
