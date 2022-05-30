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

from ..ecode import eCodes
from .efel_feature_bpem import eFELFeatureBPEM
from .protocols import BPEM_DynamicStepProtocol
from .protocols import BPEM_Protocol
from .protocols import BPEM_ThresholdProtocol
from .protocols import MainProtocol
from .protocols import RinProtocol
from .protocols import RMPProtocol
from .protocols import SearchCurrentForVoltage
from .protocols import SearchThresholdCurrent
from .recordings import FixedDtRecordingCustom
from .recordings import LooseDtRecordingCustom

logger = logging.getLogger(__name__)

soma_loc = NrnSeclistCompLocation(name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5)
ais_loc = NrnSeclistCompLocation(name="soma", seclist_name="axonal", sec_index=0, comp_x=0.5)

PRE_PROTOCOLS = [
    "SearchHoldingCurrent",
    "SearchThresholdCurrent",
    "RMPProtocol",
    "RinProtocol",
    "TRNSearchHolding",
    "TRNSearchCurrentStep",
]
LEGACY_PRE_PROTOCOLS = ["RMP", "Rin", "RinHoldcurrent", "Main", "ThresholdDetection"]

seclist_to_sec = {
    "somatic": "soma",
    "apical": "apic",
    "axonal": "axon",
    "myelinated": "myelin",
}

protocol_type_to_class = {
    "Protocol": BPEM_Protocol,
    "ThresholdBasedProtocol": BPEM_ThresholdProtocol,
    "DynamicStepProtocol": BPEM_DynamicStepProtocol,
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


def define_protocol(
    protocol_configuration,
    stochasticity=False,
    use_fixed_dt_recordings=False,
):
    """Create the protocol.

    Args:
        protocol_configuration (ProtocolConfiguration): configuration of the protocol
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        use_fixed_dt_recordings (bool): whether to record at a fixed dt of 0.1 ms.

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

        if use_fixed_dt_recordings:
            recording = FixedDtRecordingCustom(
                name=rec_def["name"],
                location=location,
                variable=variable,
            )
        else:
            recording = LooseDtRecordingCustom(
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
            f"There is no eCode linked to the stimulus name {protocol_configuration.name.lower()}. "
            "See ecode/__init__.py for the available stimuli "
            "names"
        )

    return protocol_type_to_class[protocol_configuration.protocol_type](
        name=protocol_configuration.name,
        stimulus=stimulus,
        recordings=recordings,
        stochasticity=stochasticity,
    )


def define_efeature(feature_config, protocol=None, global_efel_settings=None):
    """Define an efeature"""

    if global_efel_settings is None:
        global_efel_settings = {}

    stim_amp = None
    stim_start = None
    stim_end = None

    if protocol:
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

    efel_settings = {**global_efel_settings, **feature_config.efel_settings}
    double_settings = {k: v for k, v in efel_settings.items() if isinstance(v, float)}
    int_settings = {k: v for k, v in efel_settings.items() if isinstance(v, int)}
    string_settings = {k: v for k, v in efel_settings.items() if isinstance(v, str)}

    efeature = eFELFeatureBPEM(
        feature_config.name,
        efel_feature_name=feature_config.efel_feature_name,
        recording_names=recording_names,
        stim_start=stim_start,
        stim_end=stim_end,
        exp_mean=feature_config.mean,
        exp_std=feature_config.std,
        stimulus_current=stim_amp,
        threshold=efel_settings.get("Threshold", None),
        interp_step=efel_settings.get("interp_step", None),
        double_settings=double_settings,
        int_settings=int_settings,
        string_settings=string_settings,
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

    raise Exception(
        "Couldn't find the efeature 'steady_state_voltage_stimend' associated to "
        "the 'RMPProtocol' in your FitnessCalculatorConfiguration. It might not have"
        "been extracted from the ephys data you have available or the name of the"
        " protocol to use for RMP (setting 'name_rmp_protocol') might be wrong."
    )


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

    raise Exception(
        "Couldn't find the efeature 'ohmic_input_resistance_vb_ssse' associated to "
        "the 'RinProtocol' in your FitnessCalculatorConfiguration. It might not have"
        "been extracted from the ephys data you have available or the name of the"
        " protocol to use for Rin (setting 'name_Rin_protocol') might be wrong."
    )


def define_current_for_voltage_protocol(
    efeatures,
    protocol_name,
    target_current_name=None,
    strict_bounds=False,
    upper_bound=0.2,
    lower_bound=-0.5,
):
    """Define the search of current giving a voltage"""

    target_voltage = next(
        f
        for f in efeatures
        if protocol_name in f.recording_names[""]
        and f.efel_feature_name == "steady_state_voltage_stimend"
    )

    if target_voltage:
        return SearchCurrentForVoltage(
            name=protocol_name,
            location=soma_loc,
            target_voltage=target_voltage,
            strict_bounds=strict_bounds,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            target_current_name=target_current_name,
        )

    raise Exception(
        "Couldn't find the efeature 'steady_state_voltage_stimend' associated to "
        f"the {protocol_name} protocol in the FitnessCalculatorConfiguration."
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
        "Couldn't find the efeature 'bpo_threshold_current' or "
        "'bpo_holding_current' associated to the 'SearchHoldingCurrent'"
        " in your FitnessCalculatorConfiguration. It might not have"
        "been extracted from the ephys data you have available or the name of the"
        " protocol to use for Rin (setting 'name_Rin_protocol') might be wrong."
    )


def define_main_protocol(
    fitness_calculator_configuration,
    include_validation_protocols=False,
    stochasticity=True,
    ais_recording=False,
    efel_settings=None,
    max_threshold_voltage=-30,
    strict_holding_bounds=True,
    use_fixed_dt_recordings=False,
):
    """Create the MainProtocol and the list of efeatures to use as objectives.

    Some characteristics of the "dynamic_protocols" depend on the computation of
    the pre-protocols while the "other_protocols" do not.

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
        strict_holding_bounds (bool): to adaptively enlarge bounds is holding current is outside
        use_fixed_dt_recordings (bool): whether to record at a fixed dt of 0.1 ms.
    """

    dynamic_protocols = {}
    other_protocols = {}

    validation_protocols = [
        p.name for p in fitness_calculator_configuration.protocols if p.validation
    ]

    for protocols_def in fitness_calculator_configuration.protocols:

        if not include_validation_protocols and protocols_def.validation:
            continue

        protocol = define_protocol(protocols_def, stochasticity, use_fixed_dt_recordings)

        if isinstance(protocol, (BPEM_ThresholdProtocol, BPEM_DynamicStepProtocol)):
            dynamic_protocols[protocols_def.name] = protocol
        else:
            other_protocols[protocols_def.name] = protocol

    efeatures = []
    for feature_def in fitness_calculator_configuration.efeatures:

        if not include_validation_protocols and feature_def.protocol_name in validation_protocols:
            continue

        protocol = None
        if feature_def.protocol_name not in PRE_PROTOCOLS:
            for p in list(dynamic_protocols.values()) + list(other_protocols.values()):
                if p.name == feature_def.protocol_name:
                    protocol = p
                    break
            else:
                raise Exception(f"Could not find protocol named {feature_def.protocol_name}")

        efeatures.append(define_efeature(feature_def, protocol, efel_settings))

    pre_protocols = {}
    if any(isinstance(p, BPEM_ThresholdProtocol) for p in dynamic_protocols.values()):
        pre_protocols = {
            "rmp_protocol": define_RMP_protocol(efeatures),
            "search_holding_protocol": define_current_for_voltage_protocol(
                efeatures,
                protocol_name="SearchHoldingCurrent",
                target_current_name="bpo_holding_current",
                strict_bounds=strict_holding_bounds,
            ),
            "rin_protocol": define_Rin_protocol(efeatures, ais_recording),
            "search_threshold_protocol": define_threshold_protocol(
                efeatures, max_threshold_voltage
            ),
        }

    # TRN specific pre-protocols
    if any(isinstance(p, BPEM_DynamicStepProtocol) for p in dynamic_protocols.values()):
        pre_protocols["TRNSearchHolding"] = define_current_for_voltage_protocol(
            efeatures,
            protocol_name="TRNSearchHolding",
            target_current_name="TRNSearchHolding_current",
            strict_bounds=strict_holding_bounds,
        )
        pre_protocols["TRNSearchCurrentStep"] = define_current_for_voltage_protocol(
            efeatures,
            protocol_name="TRNSearchCurrentStep",
            target_current_name="TRNSearchCurrentStep_current",
            upper_bound=0.1,
            lower_bound=-0.4,
            strict_bounds=strict_holding_bounds,
        )

    main_protocol = MainProtocol(
        pre_protocols=pre_protocols,
        dynamic_protocols=dynamic_protocols,
        other_protocols=other_protocols,
    )

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


def get_simulator(stochasticity, cell_model, dt=None, mechanisms_directory=None, cvode_minstep=0.0):
    """Get NrnSimulator

    Args:
        stochasticity (bool): allow the use of simulator for stochastic channels
        cell_model (CellModel): used to check if any stochastic channels are present
        dt (float): if not None, cvode will be disabled and fixed timesteps used.
        mechanisms_directory (str or Path): path to the directory containing the mechanisms
        cvode_minstep (float): minimum time step allowed for a CVODE step.
    """

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

    if stochasticity:
        for mechanism in cell_model.mechanisms:
            if not mechanism.deterministic:
                return NrnSimulator(
                    dt=dt or 0.025, cvode_active=False, mechanisms_directory=mechs_parent_dir
                )

        logger.warning(
            "Stochasticity is True but no mechanisms are stochastic. Switching to "
            "non-stochastic."
        )

    if dt is None:
        return NrnSimulator(mechanisms_directory=mechs_parent_dir, cvode_minstep=cvode_minstep)

    return NrnSimulator(dt=dt, cvode_active=False, mechanisms_directory=mechs_parent_dir)


def create_evaluator(
    cell_model,
    fitness_calculator_configuration,
    include_validation_protocols=False,
    stochasticity=True,
    timeout=None,
    efel_settings=None,
    max_threshold_voltage=-30,
    dt=None,
    strict_holding_bounds=True,
    mechanisms_directory=None,
    use_fixed_dt_recordings=False,
    cvode_minstep=0.0,
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
            in the targets per efeature, the latter will have priority
        max_threshold_voltage (float): maximum voltage used as upper
            bound in the threshold current search
        dt (float): if not None, cvode will be disabled and fixed timesteps used.
        strict_holding_bounds (bool): to adaptively enlarge bounds is current is outside
        mechanisms_directory (str or Path): path to the directory containing the mechanisms
        use_fixed_dt_recordings (bool): whether to record at a fixed dt of 0.1 ms.
        cvode_minstep (float): minimum time step allowed for a CVODE step

    Returns:
        CellEvaluator
    """

    simulator = get_simulator(stochasticity, cell_model, dt, mechanisms_directory, cvode_minstep)

    fitness_calculator_configuration.configure_morphology_dependent_locations(cell_model, simulator)

    main_protocol, features = define_main_protocol(
        fitness_calculator_configuration,
        include_validation_protocols,
        stochasticity,
        efel_settings=efel_settings,
        max_threshold_voltage=max_threshold_voltage,
        strict_holding_bounds=strict_holding_bounds,
        use_fixed_dt_recordings=use_fixed_dt_recordings,
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
