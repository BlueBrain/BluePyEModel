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

filenames = [
    #"./ephys_data/ABC.nwb"
    "./ephys_data/C060109A1-SR-C1/" # if using igor files, please provide the path to the folder containing the .ibw files
]

ecodes_metadata = {
    'IDthresh': {"ljp": 14.0,"ton": 700, "toff": 2700},
    'IDrest': {"ljp": 14.0, "ton": 700, "toff": 2700},
    'IV': {"ljp": 14.0, "ton": 20, "toff": 1020},
    'APWaveform': {"ljp": 14.0, "ton": 5, "toff": 55},
    'sAHP': {"ljp": 14.0, "ton": 25, "tmid": 520, "tmid2": 720,"toff": 2720},
}


protocols_rheobase = ["IDthresh"]

targets = {
    "IDrest": {
        "amplitudes": [150, 250],
        "efeatures": [
            'Spikecount',
            'mean_frequency',
            'time_to_first_spike',
            'time_to_last_spike',
            'inv_time_to_first_spike',
            'inv_first_ISI',
            'inv_second_ISI',
            'inv_third_ISI',
            'inv_last_ISI',
            'AHP_depth',
            'AHP_time_from_peak',
            'min_AHP_values',
            'depol_block_bool',
            'voltage_base'
        ]
    },
    "IV": {
        "amplitudes": [0, -20, -100],
        "efeatures": [
            'voltage_base',
            'ohmic_input_resistance_vb_ssse',
        ]
    },
    "APWaveform": {
        "amplitudes": [280],
        "efeatures": [
            "AP_amplitude",
            "AP1_amp",
            "AP_duration_half_width",
            "AHP_depth"
        ]
    },
    "sAHP": {
        "amplitudes": [220],
        "efeatures": [
            "mean_frequency",
            "voltage_base",
            "depol_block_bool"
        ]
    }
}