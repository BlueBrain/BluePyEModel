import pytest

from bluepyemodel.efeatures_extraction.targets_configuration import TargetsConfiguration

@pytest.fixture
def config_dict():

    config_dict = {
        "files": [{
            "cell_name": "test_cell",
            "filename": "test_file",
            "ecodes": {"IDRest": {}}
        }],
        "targets": [{
                "efeature": "Spikecount",
                "protocol": "IDRest",
                "amplitude": 150.,
                "tolerance": 10.,
                "efel_settings": {"interp_step": 0.01}
            }],
        "protocols_rheobase": ["IDRest"]
    }

    return config_dict

def test_init(config_dict):

    config = TargetsConfiguration(
        files=config_dict["files"],
        targets=config_dict["targets"],
        protocols_rheobase=config_dict["protocols_rheobase"]
    )

    assert len(config.files) == 1
    assert len(config.targets) == 1
    assert len(config.as_dict()["files"]) == 1
    assert len(config.as_dict()["targets"]) == 1
    assert len(config.targets_BPE) == 1
    assert len(config.files_metadata_BPE) == 1
