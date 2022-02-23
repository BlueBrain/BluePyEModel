filenames = [
    "001_140408_A1_S1L5py_MPG",
    "001_140509_A1_S1L5py_MPG",
    "001_140514_B1_S1L5py_MPG",
    "001_140516_A1_S1L5py_MPG",
    "001_140516_B1_S1L5py_MPG",
    "001_150122_A1_S1L5py_MPG",
    "001_150122_A2_S1L5py_MPG",
    "001_150209_A2_S1L5py_MPG"
]

ecodes_metadata = {
    "IDThres": {"ljp": 14.0,},
    'IDRest': {"ljp": 14.0,},
    'IV': {"ljp": 14.0,},
    'APWaveform': {"ljp": 14.0,},
    'StartNoHold': {"ljp": 14.0,},
}

protocols_rheobase = ["IDThres"]

targets = {
    "IDRest": {
        "amplitudes": [140, 180, 260],
        "efeatures": [
            'AP_height',
            'ISI_CV',
            'doublet_ISI',
            'adaptation_index2',
            'mean_frequency',
            'AHP_depth_abs_slow',
            'AP_width',
            'time_to_first_spike',
            'AHP_depth_abs',
            'ISI_log_slope',
            'ISI_log_slope_skip',
            'voltage_base',
            'time_to_last_spike',
            'inv_time_to_first_spike',
            'inv_first_ISI',
            'inv_second_ISI',
            'inv_third_ISI',
            'inv_fourth_ISI',
            'inv_fifth_ISI',
            'inv_last_ISI',
            'AP_amplitude',
            'AP1_amp',
            'APlast_amp',
            'AP_duration_half_width',
            'AHP_depth',
            'AHP_time_from_peak',
            'min_AHP_values'
        ]
    },
    "IV": {
        "amplitudes": [-100, -40],
        "efeatures": [
            'voltage_base',
            'steady_state_voltage_stimend',
            'ohmic_input_resistance_vb_ssse',
            'voltage_deflection',
            'voltage_deflection_begin',
            'decay_time_constant_after_stim',
            'Spikecount',
            'sag_ratio1',
            'sag_ratio2',
            'sag_amplitude',
            'sag_time_constant'
        ]
    },
    "APWaveform": {
        "amplitudes": [220],
        "efeatures": [
            "AP_amplitude",
            "AP1_amp",
            "AP2_amp",
            "AP_duration_half_width",
            "AHP_depth"
        ]
    },
    "StartNoHold": {
        "amplitudes": [0],
        "efeatures": [
            "voltage_base"
        ]
    }
}
