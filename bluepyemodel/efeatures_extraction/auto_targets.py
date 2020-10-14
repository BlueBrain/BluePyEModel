"""Auto-targets-related functions."""

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

logger = logging.getLogger(__name__)


AUTO_TARGET_DICT = {
    "firing_pattern": {
        "protocols": [
            "Step",
            "FirePattern",
            "IDrest",
            "IDRest",
            "IDthresh",
            "IDThresh",
            "IDThres",
            "IDthres",
            "IV",
        ],
        "amplitudes": [200, 150, 250, 300],
        "efeatures": [
            "voltage_base",
            "adaptation_index2",
            "mean_frequency",
            "time_to_first_spike",
            "time_to_last_spike",
            "inv_first_ISI",
            "inv_second_ISI",
            "inv_third_ISI",
            "inv_fourth_ISI",
            "inv_fifth_ISI",
            "inv_last_ISI",
            "ISI_CV",
            "ISI_log_slope",
            "doublet_ISI",
            "AP_amplitude",
            "AP1_amp",
            "APlast_amp",
            "AHP_depth",
            "AHP_time_from_peak",
        ],
        "min_recordings_per_amplitude": 1,
        "preferred_number_protocols": 2,
        "tolerance": 25.0,
    },
    "ap_waveform": {
        "protocols": [
            "APWaveform",
            "APwaveform",
            "Step",
            "FirePattern",
            "IDrest",
            "IDRest",
            "IDthresh",
            "IDThresh",
            "IDThres",
            "IDthres",
            "IV",
        ],
        "amplitudes": [300, 350, 250, 400, 200],
        "efeatures": [
            "AP_amplitude",
            "AP1_amp",
            "AP2_amp",
            "AP_width",
            "AP_duration_half_width",
            "AP_rise_time",
            "AP_fall_time",
            "AHP_depth_abs",
            "AHP_time_from_peak",
            "AHP_depth",
        ],
        "min_recordings_per_amplitude": 1,
        "preferred_number_protocols": 1,
        "tolerance": 25.0,
    },
    "iv": {
        "protocols": ["IV", "Step"],
        "amplitudes": [0, -40, -100],
        "efeatures": [
            "voltage_base",
            "steady_state_voltage_stimend",
            "ohmic_input_resistance_vb_ssse",
            "voltage_deflection",
            "voltage_deflection_begin",
            "decay_time_constant_after_stim",
            "Spikecount",
            "sag_ratio1",
            "sag_ratio2",
            "sag_amplitude",
            "sag_time_constant",
        ],
        "min_recordings_per_amplitude": 1,
        "preferred_number_protocols": 2,
        "tolerance": 10.0,
    },
}


def get_auto_target_from_presets(presets):
    """Returns auto-target given preset name.

    Args:
        presets (list of str): list of preset names
    """
    auto_targets = []
    for preset in presets:
        if preset in AUTO_TARGET_DICT:
            auto_targets.append(AUTO_TARGET_DICT[preset])
        else:
            logger.warning("Could not find auto-target preset %s", preset)
    return auto_targets
