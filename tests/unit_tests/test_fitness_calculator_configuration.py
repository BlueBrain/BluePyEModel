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

import pytest

from bluepyemodel.evaluation.evaluator import get_simulator
from bluepyemodel.evaluation.fitness_calculator_configuration import (
    FitnessCalculatorConfiguration,
)
from bluepyemodel.model import model


@pytest.fixture
def config_dict():

    efeatures = [
        {
            "efel_feature_name": "voltage_base",
            "protocol_name": "Step_150",
            "recording_name": "soma.v",
            "efel_settings": {"stim_start": 700, "stim_end": 2700},
            "mean": -83.1596,
            "std": 1.0102,
            "threshold_efeature_std": 0.05,
        },
        {
            "efel_feature_name": "voltage_deflection",
            "protocol_name": "Step_150",
            "recording_name": "soma.v",
            "efel_settings": {},
            "mean": -13.5153,
            "std": 0.001,
            "threshold_efeature_std": 0.05,
        },
    ]

    protocols = [
        {
            "name": "Step_150",
            "stimuli": [
                {
                    "location": "soma",
                    "delay": 700.0,
                    "amp": None,
                    "thresh_perc": 148.7434,
                    "duration": 2000.0,
                    "totduration": 3000.0,
                    "holding_current": None,
                }
            ],
            "recordings": [
                {
                    "type": "CompRecording",
                    "name": "Step_150.soma.v",
                    "location": "soma",
                    "variable": "v",
                }
            ],
            "validation": False,
        },
        {
            "name": "Step_250",
            "stimuli": [
                {
                    "location": "soma",
                    "delay": 700.0,
                    "amp": None,
                    "thresh_perc": 250.0,
                    "duration": 2000.0,
                    "totduration": 3000.0,
                    "holding_current": None,
                }
            ],
            "recordings": [
                {
                    "type": "CompRecording",
                    "name": "Step_250.soma.v",
                    "location": "soma",
                    "variable": "v",
                }
            ],
            "validation": False,
        },
    ]

    config_dict = {
        "efeatures": efeatures,
        "protocols": protocols,
        "name_rmp_protocol": "IV_-40",
        "name_rin_protocol": "IV_0",
    }

    return config_dict


@pytest.fixture
def config_dict_bad_recordings():

    efeatures = [
        {
            "efel_feature_name": "maximum_voltage_from_voltagebase",
            "protocol_name": "Step_150",
            "recording_name": "soma.v",
            "efel_settings": {"stim_start": 700, "stim_end": 2700},
            "mean": -83.1596,
            "std": 1.0102,
            "threshold_efeature_std": 0.05,
        },
        {
            "efel_feature_name": "maximum_voltage_from_voltagebase",
            "protocol_name": "Step_150",
            "recording_name": "dend100.v",
            "efel_settings": {"stim_start": 700, "stim_end": 2700},
            "mean": -83.1596,
            "std": 1.0102,
            "threshold_efeature_std": 0.05,
        },
        {
            "efel_feature_name": "maximum_voltage_from_voltagebase",
            "protocol_name": "Step_150",
            "recording_name": "dend10000.v",
            "efel_settings": {"stim_start": 700, "stim_end": 2700},
            "mean": -83.1596,
            "std": 1.0102,
            "threshold_efeature_std": 0.05,
        },
    ]

    protocols = [
        {
            "name": "Step_150",
            "stimuli": [
                {
                    "location": "soma",
                    "delay": 700.0,
                    "amp": None,
                    "thresh_perc": 148.7434,
                    "duration": 2000.0,
                    "totduration": 3000.0,
                    "holding_current": None,
                }
            ],
            "recordings": [
                {
                    "type": "CompRecording",
                    "name": "Step_150.soma.v",
                    "location": "soma",
                    "variable": "v",
                },
                {
                    "type": "somadistanceapic",
                    "somadistance": 100,
                    "name": "Step_150.dend100.v",
                    "seclist_name": "apical",
                    "variable": "v",
                },
                {
                    "type": "somadistanceapic",
                    "somadistance": 10000,
                    "name": "Step_150.dend10000.v",
                    "seclist_name": "apical",
                    "variable": "v",
                },
            ],
            "validation": False,
        }
    ]

    config_dict = {
        "efeatures": efeatures,
        "protocols": protocols,
        "name_rmp_protocol": "IV_-40",
        "name_rin_protocol": "IV_0",
    }

    return config_dict


@pytest.fixture
def config_dict_from_bpe2():

    efeatures = {
        "Step_150": {
            "soma.v": [
                {"feature": "AP_amplitude", "val": [69.9585, 7.4575]},
            ]
        },
        "IV_-40": {
            "soma.v": [
                {
                    "feature": "ohmic_input_resistance_vb_ssse",
                    "val": [53.3741, 17.1169],
                },
                {"feature": "voltage_base", "val": [-82.9501, 0.8908]},
            ]
        },
        "IV_0": {
            "soma.v": [
                {"feature": "voltage_base", "val": [-77.1196, 3.6608]},
            ]
        },
    }

    protocols = {
        "Step_150": {
            "type": "StepThresholdProtocol",
            "step": {
                "delay": 700.0,
                "amp": None,
                "thresh_perc": 148.7434,
                "duration": 2000.0,
                "totduration": 3000.0,
            },
            "holding": {
                "delay": 0.0,
                "amp": -0.001,
                "duration": 3000.0,
                "totduration": 3000.0,
            },
        }
    }

    currents = {
        "holding_current": [-0.1573, 0.0996],
        "threshold_current": [0.3094, 0.0942],
    }

    return efeatures, protocols, currents


def test_init(config_dict):

    config = FitnessCalculatorConfiguration(**config_dict)

    assert len(config.protocols) == 2
    assert len(config.efeatures) == 2
    assert len(config.as_dict()["efeatures"]) == 2
    assert len(config.as_dict()["protocols"]) == 2

    config.remove_featureless_protocols()
    assert len(config.protocols) == 1

    keys = ["name", "stimuli", "recordings_from_config", "validation"]
    p_dict = config.protocols[0].as_dict()
    for k in keys:
        assert k in p_dict
        assert p_dict[k] is not None

    keys = [
        "efel_feature_name",
        "protocol_name",
        "efel_settings",
        "mean",
        "original_std",
    ]
    f_dict = config.efeatures[0].as_dict()
    for k in keys:
        assert k in f_dict
        assert f_dict[k] is not None

    assert config.efeatures[0].name == "Step_150.soma.v.voltage_base"
    for f in config.efeatures:
        assert f.std >= abs(0.05 * f.mean)

    # test recreation from dict output
    new_config = FitnessCalculatorConfiguration(**config.as_dict())


def test_init_from_bpe2(config_dict, config_dict_from_bpe2):

    efeatures, protocols, currents = config_dict_from_bpe2

    config_dict.pop("efeatures")
    config_dict.pop("protocols")

    config = FitnessCalculatorConfiguration(**config_dict)
    config.init_from_bluepyefe(efeatures, protocols, currents, threshold_efeature_std=0.05)

    assert len(config.protocols) == 1
    assert len(config.efeatures) == 6

    for f in config.efeatures:
        assert f.std >= abs(0.05 * f.mean)

    for fn in [
        "steady_state_voltage_stimend",
        "bpo_holding_current",
        "bpo_threshold_current",
        "ohmic_input_resistance_vb_ssse",
    ]:
        assert next((f for f in config.efeatures if f.efel_feature_name == fn), False)


def test_configure_morphology_dependent_locations(config_dict_bad_recordings, db):
    """Test configure_morphology_dependent_locations with recordings outside of cell."""
    config = FitnessCalculatorConfiguration(**config_dict_bad_recordings)

    model_configuration = db.get_model_configuration()
    cell_model = model.create_cell_model(
        name=db.emodel_metadata.emodel,
        model_configuration=model_configuration,
        morph_modifiers=None,
    )

    # don't know if I have to specify the mechanism directory here
    simulator = get_simulator(
        stochasticity=False,
        cell_model=cell_model,
    )

    config.configure_morphology_dependent_locations(cell_model, simulator)

    assert len(config.protocols) == 1
    # one recording should have been removed
    assert len(config.protocols[0].recordings) == 2
    for rec in config.protocols[0].recordings:
        if rec["type"] == "somadistanceapic":
            # check _set_morphology_dependent_locations section name replacement
            assert rec["sec_name"] == "apic"
            # check that this recording has been removed
            assert rec["somadistance"] != 10000
    # one efeature should have been removed
    assert len(config.efeatures) == 2
    for feat in config.efeatures:
        # this efeature should have been removed
        assert feat.recording_name != "dend10000.v"
