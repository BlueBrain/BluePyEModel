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

from bluepyemodel.efeatures_extraction.targets_configuration import TargetsConfiguration


@pytest.fixture
def config_dict():

    config_dict = {
        "files": [
            {
                "cell_name": "test_cell",
                "filename": "test_file",
                "ecodes": {"IDRest": {}},
            }
        ],
        "targets": [
            {
                "efeature": "Spikecount",
                "protocol": "IDRest",
                "amplitude": 150.0,
                "tolerance": 10.0,
                "efel_settings": {"interp_step": 0.01},
            }
        ],
        "protocols_rheobase": ["IDRest"],
    }

    return config_dict


def test_init(config_dict):

    config = TargetsConfiguration(
        files=config_dict["files"],
        targets=config_dict["targets"],
        protocols_rheobase=config_dict["protocols_rheobase"],
    )

    assert len(config.files) == 1
    assert len(config.targets) == 1
    assert len(config.as_dict()["files"]) == 1
    assert len(config.as_dict()["targets"]) == 1
    assert len(config.targets_BPE) == 1
    assert len(config.files_metadata_BPE) == 1
