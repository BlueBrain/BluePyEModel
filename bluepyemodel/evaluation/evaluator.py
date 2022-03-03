"""Evaluator module."""
import logging
import os
import pathlib

from bluepyopt.ephys.evaluators import CellEvaluator
from bluepyopt.ephys.locations import NrnSeclistCompLocation
from bluepyopt.ephys.locations import NrnSecSomaDistanceCompLocation
from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation
from bluepyopt.ephys.objectives import SingletonObjective
from bluepyopt.ephys.objectivescalculators import ObjectivesCalculator
from bluepyopt.ephys.simulators import NrnSimulator
from extract_currs.recordings import RecordingCustom

from ..ecode import eCodes
from .efel_feature_bpem import eFELFeatureBPEM
from .protocols import BPEM_Protocol
from .protocols import MainProtocol
from .protocols import RinProtocol
from .protocols import RMPProtocol
from .protocols import SearchHoldingCurrent
from .protocols import SearchThresholdCurrent

logger = logging.getLogger(__name__)

soma_loc = NrnSeclistCompLocation(name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5)
ais_loc = NrnSeclistCompLocation(name="soma", seclist_name="axonal", sec_index=0, comp_x=0.5)

PRE_PROTOCOLS = ["SearchHoldingCurrent", "SearchThresholdCurrent", "RMPProtocol", "RinProtocol"]
LEGACY_PRE_PROTOCOLS = ["RMP", "Rin", "RinHoldcurrent", "Main", "ThresholdDetection"]

seclist_to_sec = {
    "somatic": "soma",
    "apical": "apic",
    "axonal": "axon",
    "myelinated": "myelin",
}


def define_location(definition):

    if definition["type"] == "CompRecording":
        if definition["location"] == "soma":
            return soma_loc
        if definition["location"] == "ais":
            return ais_loc
        raise Exception("Only soma and ais are implemented for CompRecording")

    if definition["type"] == "somadistance":
        return NrnSomaDistanceCompLocation(
            name=definition["name"],
            soma_distance=definition["somadistance"],
            seclist_name=definition["seclist_name"],
        )

    if definition["type"] == "somadistanceapic":
        return NrnSecSomaDistanceCompLocation(
            name=definition["name"],
            soma_distance=definition["somadistance"],
            sec_index=definition.get("sec_index", None),
            sec_name=definition["seclist_name"],
        )

    if definition["type"] == "nrnseclistcomp":
        return NrnSeclistCompLocation(
            name=definition["name"],
            comp_x=definition["comp_x"],
            sec_index=definition["sec_index"],
            seclist_name=definition["seclist_name"],
        )

    raise Exception(f"Unknown recording type {definition['type']}")


def define_protocol(protocol_configuration, stochasticity=False, threshold_based=False):
    """Create the protocol.

    Args:
        protocol_configuration (ProtocolConfiguration): configuration of the protocol
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        threshold_based (bool): is the protocol being instantiated a threshold-based or a
            fix protocol.

    Returns:
        Protocol
    """

    recordings = []
    for rec_def in protocol_configuration.recordings:

        location = define_location(rec_def)
        if "variable" in rec_def:
            variable = rec_def["variable"]
        else:
            variable = rec_def["var"]

        recording = RecordingCustom(
            name=rec_def["name"],
            location=location,
            variable=variable,
        )

        recordings.append(recording)

    if len(protocol_configuration.stimuli) != 1:
        raise Exception("Only protocols with a single stimulus implemented")

    for k, ecode in eCodes.items():
        if k in protocol_configuration.name.lower():
            stimulus = ecode(location=soma_loc, **protocol_configuration.stimuli[0])
            break
    else:
        raise KeyError(
            f"There is no eCode linked to the stimulus name {protocol_configuration.lower()}. "
            "See ecode/__init__.py for the available stimuli "
            "names"
        )

    return BPEM_Protocol(
        name=protocol_configuration.name,
        stimulus=stimulus,
        recordings=recordings,
        stochasticity=stochasticity,
        threshold_based=threshold_based,
    )


def define_efeature(feature_config, protocol=None, global_efel_settings=None):
    """Define an efeature"""

    if global_efel_settings is None:
        global_efel_settings = {}

    efel_settings = {**global_efel_settings, **feature_config.efel_settings}

    stim_amp = None
    stim_start = None
    stim_end = None

    if protocol:
        print(f"protocol anmplitude: {protocol.amplitude}")
        stim_amp = protocol.amplitude

    if feature_config.efel_settings.get("stim_start", None) is not None:
        stim_start = feature_config.efel_settings["stim_start"]
    elif protocol:
        stim_start = protocol.stim_start

    if feature_config.efel_settings.get("stim_end", None) is not None:
        stim_end = feature_config.efel_settings["stim_end"]
    elif protocol:
        if "bAP" in protocol.name:
            stim_end = protocol.total_duration
        else:
            stim_end = protocol.stim_end

    recording_names = {"": f"{feature_config.protocol_name}.{feature_config.recording_name}"}

    efeature = eFELFeatureBPEM(
        feature_config.name,
        efel_feature_name=feature_config.efel_feature_name,
        recording_names=recording_names,
        stim_start=stim_start,
        stim_end=stim_end,
        exp_mean=feature_config.mean,
        exp_std=feature_config.std,
        stimulus_current=stim_amp,
        efel_settings=efel_settings,
    )

    return efeature


def define_RMP_protocol(efeatures):
    """Define the resting membrane potential protocol"""

    target_voltage = next(
        f
        for f in efeatures
        if (
            "RMPProtocol" in f.recording_names[""]
            and f.efel_feature_name == "steady_state_voltage_stimend"
        )
    )

    if target_voltage:

        rmp_protocol = RMPProtocol(
            name="RMPProtocol", location=soma_loc, target_voltage=target_voltage
        )

        for f in efeatures:
            if (
                "RMPProtocol" in f.recording_names[""]
                and f.efel_feature_name != "steady_state_voltage_stimend"
            ):
                f.stim_start = 0.0
                f.stim_end = rmp_protocol.stimulus_duration
                f.stimulus_current = 0.0

        return rmp_protocol

    raise Exception("Couldn't find the voltage feature associated to the RMP protocol")


def define_Rin_protocol(efeatures, ais_recording=False):
    """Define the input resistance protocol"""

    target_rin = next(
        f
        for f in efeatures
        if "RinProtocol" in f.recording_names[""]
        and f.efel_feature_name == "ohmic_input_resistance_vb_ssse"
    )

    if target_rin:

        location = soma_loc if not ais_recording else ais_loc

        return RinProtocol(name="RinProtocol", location=location, target_rin=target_rin)

    raise Exception("Couldn't find the Rin feature associated to the Rin protocol")


def define_holding_protocol(efeatures, strict_bounds=False):
    """Define the search holding current protocol"""

    target_voltage = next(
        f
        for f in efeatures
        if "SearchHoldingCurrent" in f.recording_names[""]
        and f.efel_feature_name == "steady_state_voltage_stimend"
    )
    target_current = next(
        f
        for f in efeatures
        if "SearchHoldingCurrent" in f.recording_names[""]
        and f.efel_feature_name == "bpo_holding_current"
    )

    if target_voltage and target_current:
        return SearchHoldingCurrent(
            name="SearchHoldingCurrent",
            location=soma_loc,
            target_voltage=target_voltage,
            target_holding=target_current,
            strict_bounds=strict_bounds,
        )

    raise Exception(
        "Couldn't find the voltage feature or holding current feature associated "
        "to the SearchHoldingCurrent protocol"
    )


def define_threshold_protocol(efeatures, max_threshold_voltage=-30):
    """Define the search threshold current protocol"""

    target_current = next(
        f
        for f in efeatures
        if "SearchThresholdCurrent" in f.recording_names[""]
        and f.efel_feature_name == "bpo_threshold_current"
    )

    if target_current:
        return SearchThresholdCurrent(
            name="SearchThresholdCurrent",
            location=soma_loc,
            target_threshold=target_current,
            max_threshold_voltage=max_threshold_voltage,
        )

    raise Exception(
        "Couldn't find the threshold current feature associated "
        "to the SearchThresholdCurrent protocol"
    )


def define_main_protocol(
    fitness_calculator_configuration,
    include_validation_protocols=False,
    stochasticity=True,
    ais_recording=False,
    efel_settings=None,
    score_threshold=12.0,
    max_threshold_voltage=-30,
    threshold_based_evaluator=True,
    strict_holding_bounds=True,
):
    """Create the MainProtocol and the list of efeatures to use as objectives.

    The amplitude of the "threshold_protocols" depend on the computation of
    the current threshold while the "other_protocols" do not.

    Args:
        fitness_calculator_configuration (FitnessCalculatorConfiguration): configuration of the
            fitness calculator.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        ais_recording (bool): if True all the soma recording will be at the first axonal section.
        efel_settings (dict): eFEl settings.
        threshold_efeature_std (float): if informed, compute the std as
            threshold_efeature_std * mean if std is < threshold_efeature_std * min.
        threshold_based_evaluator (bool): if True, the protocols of the evaluator will be rescaled
            by the holding and threshold current of the model.
        strict_holding_bounds (bool): to adaptively enlarge bounds is holding current is outside
    """

    threshold_protocols = {}
    other_protocols = {}

    validation_protocols = [
        p.name for p in fitness_calculator_configuration.protocols if p.validation
    ]

    for protocols_def in fitness_calculator_configuration.protocols:

        if not include_validation_protocols and protocols_def.validation:
            continue

        protocol = define_protocol(protocols_def, stochasticity, threshold_based_evaluator)

        if threshold_based_evaluator:
            threshold_protocols[protocols_def.name] = protocol
        else:
            other_protocols[protocols_def.name] = protocol

    efeatures = []
    for feature_def in fitness_calculator_configuration.efeatures:

        if not include_validation_protocols and feature_def.protocol_name in validation_protocols:
            continue

        protocol = None
        if feature_def.protocol_name not in PRE_PROTOCOLS:
            for p in list(threshold_protocols.values()) + list(other_protocols.values()):
                if p.name == feature_def.protocol_name:
                    protocol = p
                    break
            else:
                raise Exception(f"Could not find protocol named {feature_def.protocol_name}")

        print(f"f name : {feature_def.efel_feature_name}")
        print(f"prot name : {feature_def.protocol_name}")
        # print(p)
        if protocol is not None:
            print(protocol.name)
        efeatures.append(define_efeature(feature_def, protocol, efel_settings))

    rmp_protocol = None
    rin_protocol = None
    search_holding_protocol = None
    search_threshold_protocol = None
    if threshold_based_evaluator:
        rmp_protocol = define_RMP_protocol(efeatures)
        rin_protocol = define_Rin_protocol(efeatures, ais_recording)
        search_holding_protocol = define_holding_protocol(efeatures, strict_holding_bounds)
        search_threshold_protocol = define_threshold_protocol(efeatures, max_threshold_voltage)

    for feature in efeatures:
        if feature.efel_feature_name not in ["bpo_holding_current", "bpo_threshold_current"]:
            print(feature.efel_feature_name)
            print(feature.stimulus_current)

            assert feature.stim_start is not None
            assert feature.stim_end is not None
            assert feature.stimulus_current is not None

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

    # TODO: find a more elegant way to get ride of this efeature (or return the full response
    # for the SearchHoldingCurrent and SearchThresholdCurrent)
    efeatures = [
        e
        for e in efeatures
        if not (
            "SearchHoldingCurrent" in e.recording_names[""]
            and e.efel_feature_name == "steady_state_voltage_stimend"
        )
    ]

    return main_protocol, efeatures


def define_fitness_calculator(features):
    """Creates the objectives calculator.

    Args:
        features (list): list of EFeature.

    Returns:
        ObjectivesCalculator
    """

    objectives = [SingletonObjective(feat.name, feat) for feat in features]

    return ObjectivesCalculator(objectives)


def get_simulator(stochasticity, cell_model, dt=None, mechanisms_directory=None):
    """Get NrnSimulator

    Args:
        stochasticity (bool): allow the use of simulator for stochastic channels
        cell_model (CellModel): used to check if any stochastic channels are present
        dt (float): if not None, cvode will be disabled and fixed timesteps used.
        mechanisms_directory (str or Path): path to the directory containing the mechanisms
    """

    if stochasticity:
        for mechanism in cell_model.mechanisms:
            if not mechanism.deterministic:
                return NrnSimulator(dt=dt or 0.025, cvode_active=False)

        logger.warning(
            "Stochasticity is True but no mechanisms are stochastic. Switching to "
            "non-stochastic."
        )

    if mechanisms_directory is not None:
        # To avoid double loading the same mechanisms:
        cwd = pathlib.Path(os.getcwd())
        mech_dir = pathlib.Path(mechanisms_directory)
        if cwd.resolve() == mech_dir.resolve():
            mechs_parent_dir = None
        else:
            mechs_parent_dir = str(mech_dir.parents[0])
    else:
        mechs_parent_dir = None

    if dt is None:
        return NrnSimulator(mechanisms_directory=mechs_parent_dir)

    return NrnSimulator(dt=dt, cvode_active=False, mechanisms_directory=mechs_parent_dir)


def create_evaluator(
    cell_model,
    fitness_calculator_configuration,
    include_validation_protocols=False,
    stochasticity=True,
    timeout=None,
    efel_settings=None,
    score_threshold=12.0,
    max_threshold_voltage=-30,
    dt=None,
    threshold_based_evaluator=True,
    strict_holding_bounds=True,
    mechanisms_directory=None,
):
    """Creates an evaluator for a cell model/protocols/e-feature set

    Args:
        cell_model (CellModel): cell model
        fitness_calculator_configuration (FitnessCalculatorConfiguration): configuration of the
            fitness calculator.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        protocols_definition (dict): protocols and their definition
        features_definition (dict): features means and stds
        stochasticity (bool): should the stochastic channels be stochastic or
            deterministic
        timeout (float): maximum time in second during which a protocol is
            allowed to run
        efel_settings (dict): efel settings in the form
            {setting_name: setting_value}. If settings are also informed
            in the targets per efeature, the latter will have priority.
        score_threshold (float): threshold for score of protocols to stop evaluations
        max_threshold_voltage (float): maximum voltage used as upper
            bound in the threshold current search
        dt (float): if not None, cvode will be disabled and fixed timesteps used.
        threshold_based_evaluator (bool): if True, the protocols of the evaluator will be rescaled
            by the holding and threshold current of the model.
        strict_holding_bounds (bool): to adaptively enlarge bounds is current is outside
        mechanisms_directory (str or Path): path to the directory containing the mechanisms

    Returns:
        CellEvaluator
    """

    simulator = get_simulator(stochasticity, cell_model, dt, mechanisms_directory)

    fitness_calculator_configuration.configure_morphology_dependent_locations(cell_model, simulator)

    main_protocol, features = define_main_protocol(
        fitness_calculator_configuration,
        include_validation_protocols,
        stochasticity,
        efel_settings=efel_settings,
        score_threshold=score_threshold,
        max_threshold_voltage=max_threshold_voltage,
        threshold_based_evaluator=threshold_based_evaluator,
        strict_holding_bounds=strict_holding_bounds,
    )

    fitness_calculator = define_fitness_calculator(features)
    fitness_protocols = {"main_protocol": main_protocol}

    param_names = [param.name for param in cell_model.params.values() if not param.frozen]

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
