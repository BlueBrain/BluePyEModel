"""Auto-targets-related functions."""

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
            "Spikecount",
            "depol_block_bool",
            "voltage_base",
            "voltage_after_stim",
            "mean_frequency",
            "time_to_first_spike",
            "time_to_last_spike",
            "inv_time_to_first_spike",
            "inv_first_ISI",
            "inv_second_ISI",
            "inv_third_ISI",
            "inv_last_ISI",
            "ISI_CV",
            "ISI_log_slope",
            "doublet_ISI",
            "AHP_depth",
            "AHP_time_from_peak",
            "strict_burst_number",
            "strict_burst_mean_freq",
            "number_initial_spikes",
            "irregularity_index",
            "adaptation_index",
        ],
        "min_recordings_per_amplitude": 1,
        "preferred_number_protocols": 2,
        "tolerance": 20.0,
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
            "AP_duration_half_width",
            "AHP_depth",
        ],
        "min_recordings_per_amplitude": 1,
        "preferred_number_protocols": 1,
        "tolerance": 20.0,
    },
    "iv": {
        "protocols": ["IV", "Step"],
        "amplitudes": [0, -40, -100],
        "efeatures": [
            "voltage_base",
            "ohmic_input_resistance_vb_ssse",
        ],
        "min_recordings_per_amplitude": 1,
        "preferred_number_protocols": 2,
        "tolerance": 10.0,
    },
    "validation": {
        "protocols": ["IDhyperpol", "IDHyperpol"],
        "amplitudes": [150, 200],
        "efeatures": [
            "mean_frequency",
            "voltage_base",
            "depol_block_bool",
        ],
        "min_recordings_per_amplitude": 1,
        "preferred_number_protocols": 2,
        "tolerance": 20.0,
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
