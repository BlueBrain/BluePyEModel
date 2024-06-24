"""Evaluator module."""

"""
Copyright 2023-2024 Blue Brain Project / EPFL

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

import copy
import logging
import os
import pathlib

from bluepyopt.ephys.evaluators import CellEvaluator
from bluepyopt.ephys.locations import NrnSeclistCompLocation
from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation
from bluepyopt.ephys.locations import NrnTrunkSomaDistanceCompLocation
from bluepyopt.ephys.objectives import SingletonWeightObjective
from bluepyopt.ephys.objectivescalculators import ObjectivesCalculator
from bluepyopt.ephys.simulators import NrnSimulator

from ..ecode import eCodes
from ..ecode import fixed_timestep_eCodes
from ..tools.multiprotocols_efeatures_utils import get_protocol_list_from_protocol_name
from ..tools.utils import are_same_protocol
from .efel_feature_bpem import DendFitFeature
from .efel_feature_bpem import DendFitMultiProtocolsFeature
from .efel_feature_bpem import eFELFeatureBPEM
from .protocols import BPEMProtocol
from .protocols import NoHoldingCurrent
from .protocols import ProtocolRunner
from .protocols import RinProtocol
from .protocols import RMPProtocol
from .protocols import SearchHoldingCurrent
from .protocols import SearchThresholdCurrent
from .protocols import ThresholdBasedProtocol
from .recordings import FixedDtRecordingCustom
from .recordings import LooseDtRecordingCustom

logger = logging.getLogger(__name__)

soma_loc = NrnSeclistCompLocation(name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5)
ais_loc = NrnSeclistCompLocation(name="ais", seclist_name="axonal", sec_index=0, comp_x=0.5)

PRE_PROTOCOLS = [
    "SearchHoldingCurrent",
    "SearchThresholdCurrent",
    "RMPProtocol",
    "RinProtocol",
]
LEGACY_PRE_PROTOCOLS = ["RMP", "Rin", "RinHoldcurrent", "Main", "ThresholdDetection"]

seclist_to_sec = {
    "somatic": "soma",
    "apical": "apic",
    "axonal": "axon",
    "myelinated": "myelin",
}

protocol_type_to_class = {
    "Protocol": BPEMProtocol,
    "ThresholdBasedProtocol": ThresholdBasedProtocol,
    "ThresholdBasedNoHoldingProtocol": ThresholdBasedProtocol,
    "LocalThresholdBasedProtocol": ThresholdBasedProtocol,
}


def define_location(definition):
    # default case
    if definition is None or definition == "soma":
        return soma_loc

    if definition["type"] == "CompRecording":
        if definition["location"] == "soma":
            return soma_loc
        if definition["location"] == "ais":
            return ais_loc
        raise ValueError("Only soma and ais are implemented for CompRecording")

    if definition["type"] == "somadistance":
        return NrnSomaDistanceCompLocation(
            name=definition["name"],
            soma_distance=definition["somadistance"],
            seclist_name=definition["seclist_name"],
        )

    if definition["type"] == "somadistanceapic":
        return NrnTrunkSomaDistanceCompLocation(
            name=definition["name"],
            soma_distance=definition["somadistance"],
            seclist_name=definition["seclist_name"],
            direction="radial",
        )

    if definition["type"] == "nrnseclistcomp":
        return NrnSeclistCompLocation(
            name=definition["name"],
            comp_x=definition["comp_x"],
            sec_index=definition["sec_index"],
            seclist_name=definition["seclist_name"],
        )

    raise ValueError(f"Unknown location type {definition['type']}")


def define_recording(recording_conf, use_fixed_dt_recordings=False):
    """Create a recording from a configuration dictionary

    Args:
        recording_conf (dict): configuration of the recording. Must contain the type of the
            recording as well as information about the location of the recording (see function
            define_location).
        use_fixed_dt_recordings (bool): used for legacy currentscape
            to record at a fixed dt of 0.1 ms.

    Returns:
        FixedDtRecordingCustom or LooseDtRecordingCustom
    """

    location = define_location(recording_conf)
    variable = recording_conf.get("variable", recording_conf.get("var"))

    if use_fixed_dt_recordings:
        rec_class = FixedDtRecordingCustom
    else:
        rec_class = LooseDtRecordingCustom

    return rec_class(name=recording_conf["name"], location=location, variable=variable)


def define_protocol(
    protocol_configuration, stochasticity=False, use_fixed_dt_recordings=False, dt=None
):
    """Create a protocol.

    Args:
        protocol_configuration (ProtocolConfiguration): configuration of the protocol
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        use_fixed_dt_recordings (bool): used for legacy currentscape to record at a fixed dt
            of 0.1 ms. However, Simulations will be run based on dt.
        dt (float): NEURON dt for fixed time step simulation. If None, variable
            dt will be used.
    Returns:
        Protocol
    """
    cvode_active = not dt

    recordings = []
    for rec_conf in protocol_configuration.recordings:
        recordings.append(define_recording(rec_conf, use_fixed_dt_recordings))

    if len(protocol_configuration.stimuli) != 1:
        raise ValueError("Only protocols with a single stimulus implemented")

    stim = protocol_configuration.stimuli[0]
    location_dict = stim.get("location", "soma")
    location = define_location(location_dict)
    # cannot use pop here because stim["location"] is used later
    filtered_stim = {k: v for k, v in stim.items() if k != "location"}
    for k, ecode in eCodes.items():
        if k in protocol_configuration.name.lower():
            stimulus = ecode(location=location, **filtered_stim)
            if k in fixed_timestep_eCodes:
                cvode_active = False
            break
    else:
        raise KeyError(
            f"There is no eCode linked to the stimulus name {protocol_configuration.name.lower()}. "
            "See ecode/__init__.py for the available stimuli "
            "names"
        )

    stoch = stochasticity and protocol_configuration.stochasticity
    if stoch:
        cvode_active = False

    kwargs = {}
    if protocol_configuration.protocol_type == "LocalThresholdBasedProtocol":
        loc_name = location_dict["name"]
        kwargs = {
            "hold_key": f"bpo_holding_current_{loc_name}",
            "thres_key": f"bpo_threshold_current_{loc_name}",
            "hold_prot_name": f"SearchHoldingCurrent_{loc_name}",
            "thres_prot_name": f"SearchThresholdCurrent_{loc_name}",
        }

    if protocol_configuration.protocol_type == "ThresholdBasedNoHoldingProtocol":
        kwargs = {
            "hold_key": "bpo_holding_current_noholding",
            "thres_key": "bpo_threshold_current_noholding",
            "hold_prot_name": "SearchHoldingCurrent_noholding",
            "thres_prot_name": "SearchThresholdCurrent_noholding",
        }

    if protocol_configuration.protocol_type in protocol_type_to_class:
        return protocol_type_to_class[protocol_configuration.protocol_type](
            name=protocol_configuration.name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=cvode_active,
            stochasticity=stoch,
            **kwargs,
        )

    raise ValueError(f"Protocol type {protocol_configuration.protocol_type} not found")


def define_efeature(feature_config, protocol=None, global_efel_settings=None):
    """Define an efeature from a configuration dictionary"""

    global_efel_settings = {} if global_efel_settings is None else global_efel_settings

    stim_amp = None
    stim_start = None
    stim_end = None

    if protocol:
        stim_amp = protocol.amplitude

    if feature_config.efel_settings.get("stim_start", None) is not None:
        stim_start = feature_config.efel_settings["stim_start"]
    elif protocol:
        stim_start = protocol.stim_start()

    if feature_config.efel_settings.get("stim_end", None) is not None:
        stim_end = feature_config.efel_settings["stim_end"]
    elif protocol:
        if "bAP" in protocol.name:
            stim_end = protocol.total_duration
        else:
            stim_end = protocol.stim_end()

    efel_settings = {**global_efel_settings, **feature_config.efel_settings}

    # Handle the special case of multiple_decay_time_constant_after_stim
    if feature_config.efel_feature_name == "multiple_decay_time_constant_after_stim":
        if hasattr(protocol.stimuli[0], "multi_stim_start"):
            efel_settings["multi_stim_start"] = protocol.stimuli[0].multi_stim_start()
            efel_settings["multi_stim_end"] = protocol.stimuli[0].multi_stim_end()
        else:
            efel_settings["multi_stim_start"] = [stim_start]
            efel_settings["multi_stim_end"] = [stim_end]
    double_settings = {k: v for k, v in efel_settings.items() if isinstance(v, (float, list))}
    int_settings = {k: v for k, v in efel_settings.items() if isinstance(v, int)}
    string_settings = {k: v for k, v in efel_settings.items() if isinstance(v, str)}

    if "dendrite_backpropagation_fit" in feature_config.name:
        decay = "decay" in feature_config.name
        linear = "linear" in feature_config.name
        efeature = DendFitFeature(
            feature_config.name,
            efel_feature_name=feature_config.efel_feature_name,
            recording_names=feature_config.recording_name_for_instantiation,
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
            decay=decay,
            linear=linear,
            weight=feature_config.weight,
        )
    # protocol name contains list of location, e.g. 'apic[050,100,150]'
    elif "[" in feature_config.recording_name_for_instantiation[""]:
        decay = "decay" in feature_config.name
        linear = "linear" in feature_config.name
        efeature = DendFitMultiProtocolsFeature(
            feature_config.name,
            efel_feature_name=feature_config.efel_feature_name,
            recording_names=feature_config.recording_name_for_instantiation,
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
            decay=decay,
            linear=linear,
            weight=feature_config.weight,
        )
    else:
        efeature = eFELFeatureBPEM(
            feature_config.name,
            efel_feature_name=feature_config.efel_feature_name,
            recording_names=feature_config.recording_name_for_instantiation,
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
            weight=feature_config.weight,
        )

    return efeature


def define_RMP_protocol(
    efeatures,
    stimulus_duration=500.0,
    location=soma_loc,
    output_key="bpo_rmp",
    rmp_prot_name="RMPProtocol",
    recording_name=None,
):
    """Define the resting membrane potential protocol"""

    target_voltage = None
    for f in efeatures:
        if (
            "RMPProtocol" in f.recording_names[""]
            and f.efel_feature_name == "steady_state_voltage_stimend"
        ):
            if recording_name is not None:
                target_voltage = copy.deepcopy(f)
                target_voltage.recording_names = {"": f"{rmp_prot_name}.{recording_name}"}
            else:
                target_voltage = f
            break

    if not target_voltage:
        raise ValueError(
            "Couldn't find the efeature 'steady_state_voltage_stimend' associated to the "
            "'RMPProtocol' in your FitnessCalculatorConfiguration. It might not have been "
            "extracted from the ephys data you have available or the name of the protocol to"
            " use for RMP (setting 'name_rmp_protocol') might be wrong."
        )

    rmp_protocol = RMPProtocol(
        name=rmp_prot_name,
        location=location,
        target_voltage=target_voltage,
        stimulus_duration=stimulus_duration,
        output_key=output_key,
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


def define_Rin_protocol(
    efeatures,
    amp=-0.02,
    stimulus_delay=500.0,
    stimulus_duration=500.0,
    totduration=1000.0,
    location=soma_loc,
    output_key="bpo_rin",
    hold_key="bpo_holding_current",
    hold_prot_name="SearchHoldingCurrent",
    rin_prot_name="RinProtocol",
    recording_name=None,
):
    """Define the input resistance protocol"""

    target_rin = None
    for f in efeatures:
        if (
            "RinProtocol" in f.recording_names[""]
            and f.efel_feature_name == "ohmic_input_resistance_vb_ssse"
        ):
            if recording_name is not None:
                target_rin = copy.deepcopy(f)
                target_rin.recording_names = {"": f"{rin_prot_name}.{recording_name}"}
            else:
                target_rin = f
            break

    if not target_rin:
        raise ValueError(
            "Couldn't find the efeature 'ohmic_input_resistance_vb_ssse' associated to "
            "the 'RinProtocol' in your FitnessCalculatorConfiguration. It might not have"
            "been extracted from the ephys data you have available or the name of the"
            " protocol to use for Rin (setting 'name_Rin_protocol') might be wrong."
        )

    return RinProtocol(
        name=rin_prot_name,
        location=location,
        target_rin=target_rin,
        amp=amp,
        stimulus_delay=stimulus_delay,
        stimulus_duration=stimulus_duration,
        totduration=totduration,
        output_key=output_key,
        hold_key=hold_key,
        hold_prot_name=hold_prot_name,
    )


def define_holding_protocol(
    efeatures,
    strict_bounds=False,
    max_depth=7,
    stimulus_duration=500.0,
    location=soma_loc,
    output_key="bpo_holding_current",
    hold_prot_name="SearchHoldingCurrent",
    recording_name=None,
):
    """Define the search holding current protocol"""

    target_voltage = None
    for f in efeatures:
        if (
            "SearchHoldingCurrent" in f.recording_names[""]
            and f.efel_feature_name == "steady_state_voltage_stimend"
        ):
            if recording_name is not None:
                target_voltage = copy.deepcopy(f)
                target_voltage.recording_names = {"": f"{hold_prot_name}.{recording_name}"}
            else:
                target_voltage = f
            break

    if target_voltage:
        return SearchHoldingCurrent(
            name=hold_prot_name,
            location=location,
            target_voltage=target_voltage,
            strict_bounds=strict_bounds,
            max_depth=max_depth,
            stimulus_duration=stimulus_duration,
            output_key=output_key,
        )

    raise ValueError(
        "Couldn't find the efeature 'bpo_holding_current' associated to "
        "the 'SearchHoldingCurrent' protocol in your FitnessCalculatorConfiguration."
    )


def define_threshold_protocol(
    efeatures,
    max_threshold_voltage=-30,
    step_delay=500.0,
    step_duration=2000.0,
    totduration=3000.0,
    spikecount_timeout=50,
    max_depth=10,
    location=soma_loc,
    no_spikes=True,
    efel_threshold=None,
    output_key="bpo_threshold_current",
    hold_key="bpo_holding_current",
    rmp_key="bpo_rmp",
    rin_key="bpo_rin",
    hold_prot_name="SearchHoldingCurrent",
    rmp_prot_name="RMPProtocol",
    rin_prot_name="RinProtocol",
    thres_prot_name="SearchThresholdCurrent",
):
    """Define the search threshold current protocol"""
    # pylint: disable=too-many-arguments

    target_current = []
    for f in efeatures:
        if thres_prot_name in f.recording_names[""] and f.efel_feature_name == output_key:
            target_current.append(f)

    if not target_current:
        target_current = [None]

    return SearchThresholdCurrent(
        name=thres_prot_name,
        location=location,
        target_threshold=target_current[0],
        max_threshold_voltage=max_threshold_voltage,
        stimulus_delay=step_delay,
        stimulus_duration=step_duration,
        stimulus_totduration=totduration,
        spikecount_timeout=spikecount_timeout,
        max_depth=max_depth,
        no_spikes=no_spikes,
        efel_threshold=efel_threshold,
        output_key=output_key,
        hold_key=hold_key,
        rmp_key=rmp_key,
        rin_key=rin_key,
        hold_prot_name=hold_prot_name,
        rmp_prot_name=rmp_prot_name,
        rin_prot_name=rin_prot_name,
    )


def define_protocols(
    fitness_calculator_configuration,
    include_validation_protocols,
    stochasticity,
    use_fixed_dt_recordings,
    dt,
):
    """Instantiate several efeatures"""

    protocols = {}
    for protocols_def in fitness_calculator_configuration.protocols:
        if not include_validation_protocols and protocols_def.validation:
            continue
        protocols[protocols_def.name] = define_protocol(
            protocols_def, stochasticity, use_fixed_dt_recordings, dt
        )

    return protocols


def define_efeatures(
    fitness_calculator_configuration,
    include_validation_protocols,
    protocols,
    efel_settings,
):
    """Instantiate several Protocols"""

    efeatures = []
    validation_prot = fitness_calculator_configuration.validation_protocols

    for feature_def in fitness_calculator_configuration.efeatures:
        if not include_validation_protocols and any(
            are_same_protocol(feature_def.protocol_name, p) for p in validation_prot
        ):
            continue

        protocol = None
        # split("_")[0] is to account for 'local' pre-protocol,
        # e.g. "SearchThresholdCurrent_apical055"
        if (
            feature_def.protocol_name not in PRE_PROTOCOLS
            and feature_def.protocol_name.split("_")[0] not in PRE_PROTOCOLS
        ):
            if "[" in feature_def.protocol_name:
                prot_names = get_protocol_list_from_protocol_name(feature_def.protocol_name)
            else:
                prot_names = [feature_def.protocol_name]
            for prot_name in prot_names:
                # here, we assume that all protocols used in a DendFitMultiProtocolsFeature
                # have the same stim_start and stim_end, and also that
                # we will not use any feature depending on protocol stimulus current.
                # As such, we can pass any protocol matching any item in prot_names.
                protocol = next((p for p in protocols.values() if p.name == prot_name), None)
                if protocol is None:
                    raise ValueError(f"Could not find protocol named {prot_name}")

        efeatures.append(define_efeature(feature_def, protocol, efel_settings))

    return efeatures


def define_optimisation_protocol(
    fitness_calculator_configuration,
    include_validation_protocols=False,
    stochasticity=True,
    efel_settings=None,
    use_fixed_dt_recordings=False,
    dt=None,
):
    """Create a meta protocol in charge of running the other protocols.

    Args:
        fitness_calculator_configuration (FitnessCalculatorConfiguration): configuration of the
            fitness calculator.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        efel_settings (dict): eFEl settings.
        use_fixed_dt_recordings (bool): used for legacy currentscape
            to record at a fixed dt of 0.1 ms.
        dt (float): NEURON dt for fixed time step simulation. If None, variable dt will be used.
    """

    protocols = define_protocols(
        fitness_calculator_configuration,
        include_validation_protocols,
        stochasticity,
        use_fixed_dt_recordings,
        dt,
    )

    efeatures = define_efeatures(
        fitness_calculator_configuration,
        include_validation_protocols,
        protocols,
        efel_settings,
    )

    protocol_runner = ProtocolRunner(protocols)

    return protocol_runner, efeatures


def define_preprotocols(
    efeatures,
    location,
    fitness_calculator_configuration,
    strict_holding_bounds=False,
    max_depth_holding_search=7,
    max_threshold_voltage=-30,
    spikecount_timeout=50,
    max_depth_threshold_search=10,
    efel_thresh_for_thres_search=None,
    rmp_key="bpo_rmp",
    hold_key="bpo_holding_current",
    rin_key="bpo_rin",
    thres_key="bpo_threshold_current",
    rmp_prot_name="RMPProtocol",
    hold_prot_name="SearchHoldingCurrent",
    rin_prot_name="RinProtocol",
    thres_prot_name="SearchThresholdCurrent",
    recording_name=None,
    no_holding=False,
):
    """Returns a dict containing the preprotocols"""
    # pylint: disable=too-many-arguments
    rmp_prot = define_RMP_protocol(
        efeatures,
        stimulus_duration=fitness_calculator_configuration.rmp_duration,
        location=location,
        output_key=rmp_key,
        rmp_prot_name=rmp_prot_name,
        recording_name=recording_name,
    )
    if no_holding:
        holding_prot = NoHoldingCurrent(
            name=hold_prot_name,
            output_key=hold_key,
        )
        # without holding current, cell can spike spontaneously
        no_spikes = False
    else:
        holding_prot = define_holding_protocol(
            efeatures,
            strict_holding_bounds,
            max_depth_holding_search,
            stimulus_duration=fitness_calculator_configuration.search_holding_duration,
            location=location,
            output_key=hold_key,
            hold_prot_name=hold_prot_name,
            recording_name=recording_name,
        )
        no_spikes = True
    rin_prot = define_Rin_protocol(
        efeatures,
        amp=fitness_calculator_configuration.rin_step_amp,
        stimulus_delay=fitness_calculator_configuration.rin_step_delay,
        stimulus_duration=fitness_calculator_configuration.rin_step_duration,
        totduration=fitness_calculator_configuration.rin_totduration,
        location=location,
        output_key=rin_key,
        hold_key=hold_key,
        hold_prot_name=hold_prot_name,
        rin_prot_name=rin_prot_name,
        recording_name=recording_name,
    )
    threshold_prot = define_threshold_protocol(
        efeatures,
        max_threshold_voltage,
        fitness_calculator_configuration.search_threshold_step_delay,
        fitness_calculator_configuration.search_threshold_step_duration,
        fitness_calculator_configuration.search_threshold_totduration,
        spikecount_timeout,
        max_depth_threshold_search,
        location=location,
        no_spikes=no_spikes,
        efel_threshold=efel_thresh_for_thres_search,
        output_key=thres_key,
        hold_key=hold_key,
        rmp_key=rmp_key,
        rin_key=rin_key,
        hold_prot_name=hold_prot_name,
        rmp_prot_name=rmp_prot_name,
        rin_prot_name=rin_prot_name,
        thres_prot_name=thres_prot_name,
    )
    return {
        rmp_prot_name: rmp_prot,
        hold_prot_name: holding_prot,
        rin_prot_name: rin_prot,
        thres_prot_name: threshold_prot,
    }


def define_threshold_based_optimisation_protocol(
    fitness_calculator_configuration,
    include_validation_protocols=False,
    stochasticity=True,
    ais_recording=False,
    efel_settings=None,
    max_threshold_voltage=-30,
    strict_holding_bounds=True,
    use_fixed_dt_recordings=False,
    max_depth_holding_search=7,
    max_depth_threshold_search=10,
    spikecount_timeout=50,
    efel_threshold_for_threshold_search=None,
    dt=None,
):
    """Create a meta protocol in charge of running the other protocols.

    The amplitude of the "threshold_protocols" depend on the computation of
    the current threshold.

    Args:
        fitness_calculator_configuration (FitnessCalculatorConfiguration): configuration of the
            fitness calculator.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        ais_recording (bool): if True all the soma recording will be at the first axonal section.
        efel_settings (dict): eFEl settings.
        max_threshold_voltage (float): maximum voltage at which the SearchThresholdProtocol
            will search for the rheobase.
        strict_holding_bounds (bool): to adaptively enlarge bounds if holding current is outside
            when set to False
        use_fixed_dt_recordings (bool): used for legacy currentscape
            to record at a fixed dt of 0.1 ms.
        max_depth_holding_search (int): maximum depth for the binary search for the
            holding current
        max_depth_threshold_search (int): maximum depth for the binary search for the
            threshold current
        spikecount_timeout (float): timeout for spikecount computation, if timeout is reached,
            we set spikecount=2 as if many spikes were present, to speed up bisection search.
        efel_threshold_for_threshold_search: spike threshold for the efel settings
            to use in SearchThresholdCurrent protocol. Set to None to keep the default value
            (currently -20 mV in efel)
        dt (float): NEURON dt for fixed time step simulation. If None, variable dt will be used.
    """

    protocols = define_protocols(
        fitness_calculator_configuration,
        include_validation_protocols,
        stochasticity,
        use_fixed_dt_recordings,
        dt,
    )

    efeatures = define_efeatures(
        fitness_calculator_configuration,
        include_validation_protocols,
        protocols,
        efel_settings,
    )

    # Create the special protocols for local stimulation
    for prot in fitness_calculator_configuration.protocols:
        if include_validation_protocols or not prot.validation:
            if prot.protocol_type == "LocalThresholdBasedProtocol":
                stim = prot.stimuli[0]
                loc_name = stim["location"]["name"]
                location = define_location(stim["location"])

                if f"RMPProtocol_{loc_name}" not in protocols:
                    protocols.update(
                        define_preprotocols(
                            efeatures=efeatures,
                            location=location,
                            fitness_calculator_configuration=fitness_calculator_configuration,
                            strict_holding_bounds=strict_holding_bounds,
                            max_depth_holding_search=max_depth_holding_search,
                            max_threshold_voltage=max_threshold_voltage,
                            spikecount_timeout=spikecount_timeout,
                            max_depth_threshold_search=max_depth_threshold_search,
                            efel_thresh_for_thres_search=efel_threshold_for_threshold_search,
                            rmp_key=f"bpo_rmp_{loc_name}",
                            hold_key=f"bpo_holding_current_{loc_name}",
                            rin_key=f"bpo_rin_{loc_name}",
                            thres_key=f"bpo_threshold_current_{loc_name}",
                            rmp_prot_name=f"RMPProtocol_{loc_name}",
                            hold_prot_name=f"SearchHoldingCurrent_{loc_name}",
                            rin_prot_name=f"RinProtocol_{loc_name}",
                            thres_prot_name=f"SearchThresholdCurrent_{loc_name}",
                            recording_name=f"{loc_name}.{stim['location']['variable']}",
                            # use holding current = 0 for local injection protocols
                            # TODO: let the user decide whether to use noholding or holding
                            no_holding=True,
                        )
                    )

            if prot.protocol_type == "ThresholdBasedNoHoldingProtocol":
                stim = prot.stimuli[0]
                location = ais_loc if ais_recording else soma_loc

                if "RMPProtocol_noholding" not in protocols:
                    protocols.update(
                        define_preprotocols(
                            efeatures=efeatures,
                            location=location,
                            fitness_calculator_configuration=fitness_calculator_configuration,
                            strict_holding_bounds=strict_holding_bounds,
                            max_depth_holding_search=max_depth_holding_search,
                            max_threshold_voltage=max_threshold_voltage,
                            spikecount_timeout=spikecount_timeout,
                            max_depth_threshold_search=max_depth_threshold_search,
                            efel_thresh_for_thres_search=efel_threshold_for_threshold_search,
                            rmp_key="bpo_rmp_noholding",
                            hold_key="bpo_holding_current_noholding",
                            rin_key="bpo_rin_noholding",
                            thres_key="bpo_threshold_current_noholding",
                            rmp_prot_name="RMPProtocol_noholding",
                            hold_prot_name="SearchHoldingCurrent_noholding",
                            rin_prot_name="RinProtocol_noholding",
                            thres_prot_name="SearchThresholdCurrent_noholding",
                            recording_name="soma.v",
                            no_holding=True,
                        )
                    )

            # Create the special protocols for soma stimulation
            if prot.protocol_type == "ThresholdBasedProtocol" and "RMPProtocol" not in protocols:
                location = ais_loc if ais_recording else soma_loc
                protocols.update(
                    define_preprotocols(
                        efeatures=efeatures,
                        location=location,
                        fitness_calculator_configuration=fitness_calculator_configuration,
                        strict_holding_bounds=strict_holding_bounds,
                        max_depth_holding_search=max_depth_holding_search,
                        max_threshold_voltage=max_threshold_voltage,
                        spikecount_timeout=spikecount_timeout,
                        max_depth_threshold_search=max_depth_threshold_search,
                        efel_thresh_for_thres_search=efel_threshold_for_threshold_search,
                        no_holding=False,
                    )
                )

    # Create the protocol runner
    protocol_runner = ProtocolRunner(protocols)

    return protocol_runner, efeatures


def define_fitness_calculator(features):
    """Creates the objectives calculator.

    Args:
        features (list): list of EFeature.

    Returns:
        ObjectivesCalculator
    """

    objectives = [SingletonWeightObjective(feat.name, feat, feat.weight) for feat in features]

    return ObjectivesCalculator(objectives)


def get_simulator(stochasticity, cell_model, dt=None, mechanisms_directory=None, cvode_minstep=0.0):
    """Get NrnSimulator

    Args:
        stochasticity (bool): allow the use of simulator for stochastic channels
        cell_model (CellModel): used to check if any stochastic channels are present
        dt (float): NEURON dt for fixed time step simulation. If None, variable dt will be used.
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
                    dt=dt or 0.025,
                    cvode_active=False,
                    mechanisms_directory=mechs_parent_dir,
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
    pipeline_settings,
    stochasticity=None,
    timeout=None,
    include_validation_protocols=False,
    mechanisms_directory=None,
    use_fixed_dt_recordings=False,
):
    """Creates an evaluator for a cell model/protocols/e-feature combo.

    Args:
        cell_model (CellModel): cell model
        fitness_calculator_configuration (FitnessCalculatorConfiguration): configuration of the
            fitness calculator.
        pipeline_settings (EModelPipelineSettings): settings for the pipeline.
        stochasticity (bool): should the stochastic channels be stochastic or
            deterministic
        timeout (float): maximum time in second during which a protocol is
            allowed to run
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        mechanisms_directory (str or Path): path to the directory containing the mechanisms.
        use_fixed_dt_recordings (bool): used for legacy currentscape
            to record at a fixed dt of 0.1 ms.

    Returns:
        CellEvaluator
    """

    stochasticity = pipeline_settings.stochasticity if stochasticity is None else stochasticity
    timeout = pipeline_settings.optimisation_timeout if timeout is None else timeout

    simulator = get_simulator(
        stochasticity=stochasticity,
        cell_model=cell_model,
        dt=pipeline_settings.neuron_dt,
        mechanisms_directory=mechanisms_directory,
        cvode_minstep=pipeline_settings.cvode_minstep,
    )

    fitness_calculator_configuration.configure_morphology_dependent_locations(cell_model, simulator)

    if any(
        p.protocol_type
        in (
            "ThresholdBasedProtocol",
            "LocalThresholdBasedProtocol",
            "ThresholdBasedNoHoldingProtocol",
        )
        for p in fitness_calculator_configuration.protocols
    ):
        main_protocol, features = define_threshold_based_optimisation_protocol(
            fitness_calculator_configuration,
            include_validation_protocols,
            stochasticity=stochasticity,
            efel_settings=pipeline_settings.efel_settings,
            max_threshold_voltage=pipeline_settings.max_threshold_voltage,
            strict_holding_bounds=pipeline_settings.strict_holding_bounds,
            use_fixed_dt_recordings=use_fixed_dt_recordings,
            max_depth_holding_search=pipeline_settings.max_depth_holding_search,
            max_depth_threshold_search=pipeline_settings.max_depth_threshold_search,
            spikecount_timeout=pipeline_settings.spikecount_timeout,
            efel_threshold_for_threshold_search=pipeline_settings.efel_settings.get(
                "Threshold", None
            ),
            dt=pipeline_settings.neuron_dt,
        )
    else:
        main_protocol, features = define_optimisation_protocol(
            fitness_calculator_configuration,
            include_validation_protocols,
            stochasticity=stochasticity,
            efel_settings=pipeline_settings.efel_settings,
            use_fixed_dt_recordings=use_fixed_dt_recordings,
            dt=pipeline_settings.neuron_dt,
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
        use_params_for_seed=pipeline_settings.use_params_for_seed,
        timeout=timeout,
    )
    cell_eval.prefix = cell_model.name

    return cell_eval


def add_recordings_to_evaluator(cell_evaluator, vars, use_fixed_dt_recordings=False):
    """Add a recording for each new variable for each protocol in cell evaluator."""
    # add recording for each protocol x new variable combination
    for prot in cell_evaluator.fitness_protocols["main_protocol"].protocols.values():
        if prot.name == "SearchThresholdCurrent" or prot.name not in PRE_PROTOCOLS:
            base_rec = prot.recordings[0]
            for var in vars:
                location = base_rec.location

                split_name = base_rec.name.split(".")
                split_name[-1] = var
                name = ".".join(split_name)

                # FixedDtRecordingCustom for fixed time steps.
                # Use LooseDtRecordingCustom for variable time steps
                if use_fixed_dt_recordings:
                    new_rec = FixedDtRecordingCustom(name=name, location=location, variable=var)
                else:
                    new_rec = LooseDtRecordingCustom(name=name, location=location, variable=var)
                prot.recordings.append(new_rec)
