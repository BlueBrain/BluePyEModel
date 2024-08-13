"""
Copyright 2024, EPFL/Blue Brain Project

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



filenames = [
    "C060116A6-SR-C1",
    "C060112A4-SR-C1",
    "C060116A1-SR-C1",
]

ecodes_metadata = {
    "IDthresh": {
        "ljp": 14.0,
    },
    "IDrest": {
        "ljp": 14.0,
    },
    "IV": {"ljp": 14.0, "ton": 20, "toff": 1020},
    "APWaveform": {
        "ljp": 14.0,
    },
    "IDhyperpol": {"ljp": 14.0, "ton": 100, "tmid": 700, "tmid2": 2700, "toff": 2900},
}


protocols_rheobase = ["IDthresh"]

targets = {
    "IDrest": {
        "amplitudes": [130, 200],
        "efeatures": [
            "Spikecount",
            "ISI_CV",
            "doublet_ISI",
            "mean_frequency",
            "time_to_first_spike",
            "ISI_log_slope",
            "voltage_base",
            "time_to_last_spike",
            "inv_time_to_first_spike",
            "inv_first_ISI",
            "inv_second_ISI",
            "inv_third_ISI",
            "inv_last_ISI",
            "AHP_depth",
            "AHP_time_from_peak",
            "min_AHP_values",
            "depol_block_bool",
            "voltage_after_stim",
            "burst_number",
            "number_initial_spikes",
            "irregularity_index",
            "adaptation_index",
            "burst_mean_freq",
        ],
    },
    "IV": {
        "amplitudes": [0, -40, -100],
        "efeatures": [
            "voltage_base",
            "ohmic_input_resistance_vb_ssse",
        ],
    },
    "APWaveform": {
        "amplitudes": [280],
        "efeatures": ["AP_amplitude", "AP1_amp", "AP_duration_half_width", "AHP_depth"],
    },
    "IDhyperpol": {
        "amplitudes": [150],
        "efeatures": ["mean_frequency", "voltage_base", "depol_block_bool"],
    },
}
