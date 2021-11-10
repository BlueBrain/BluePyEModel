import pytest
from pathlib import Path
import json

from bluepyemodel.access_point import get_access_point
from dictdiffer import diff


TEST_ROOT = Path(__file__).parent
DATA = TEST_ROOT / "test_data"


@pytest.fixture
def api_config():
    return {
        "emodel": "cADpyr_L5TPC",
        "emodel_dir": DATA,
        "recipes_path": DATA / "config/recipes.json",
    }


@pytest.fixture
def db(api_config):
    return get_access_point("local", **api_config)


def test_get_morphologies(db):
    morphology = db.get_morphologies()
    assert morphology["name"] == "C060114A5"
    assert Path(morphology["path"]).name == "C060114A5.asc"


def test_get_available_morphologies(db):
    names = db.get_available_morphologies()
    assert len(names) == 1
    assert list(names)[0] == "C060114A5"


def test_get_features(db):
    features = db.get_features()
    # json.dump(features, open(DATA / "test_features.json", "w"))
    expected_features = json.load(open(DATA / "test_features.json", "r"))
    assert list(diff(features, expected_features)) == []


def test_get_protocols(db):
    protocols = db.get_protocols()
    # json.dump(protocols, open(DATA / "test_protocols.json", "w"))
    expected_protocols = json.load(open(DATA / "test_protocols.json", "r"))
    assert list(diff(protocols, expected_protocols)) == []


def test_get_recipes(db):
    recipes = db.get_recipes()
    # json.dump(recipes, open(DATA / "test_recipes.json", "w"))
    expected_recipes = json.load(open(DATA / "test_recipes.json", "r"))
    assert list(diff(recipes, expected_recipes)) == []


def test_get_model_configuration(db):

    configuration = db.get_model_configuration()

    expected_parameters = json.load(open(DATA / "test_parameters.json", "r"))
    expected_mechanisms = json.load(open(DATA / "test_mechanisms.json", "r"))

    for p in configuration.parameters:
        assert p.location in expected_parameters['parameters']
        for ep in expected_parameters['parameters'][p.location]:
            if ep['name'] == p.name and ep['val'] == p.value:
                break
        else:
            raise Exception("missing parameter")

    assert sorted(list(configuration.mechanism_names)) == [
        "CaDynamics_DC0",
        "Ca_HVA2",
        "Ca_LVAst",
        "Ih",
        "K_Pst",
        "K_Tst",
        "NaTg",
        "Nap_Et2",
        "SK_E2",
        "SKv3_1",
        "pas",
    ]


def test_get_final(db):
    final = db.get_final()
    assert "cADpyr_L5TPC" in final
    assert "params" in final["cADpyr_L5TPC"]


def test_load_pipeline_settings(db):
    assert db.pipeline_settings.path_extract_config == "tests/test_data/config/config_dict.json"
    assert db.pipeline_settings.validation_protocols == {"APWaveform": [140]}


def test_get_extraction_metadata(db):
    path_extract_config = DATA / "config" / "config_dict.json"
    with open(path_extract_config, "r") as f:
        config_dict = json.load(f)

    expected_files_metadata = config_dict["files_metadata"]
    expected_targets = config_dict["targets"]
    expected_protocols_threshold = config_dict["protocols_threshold"]

    (files_metadata, targets, protocols_threshold) = db.get_extraction_metadata()
    assert list(diff(files_metadata, expected_files_metadata)) == []
    assert list(diff(targets, expected_targets)) == []
    assert list(diff(protocols_threshold, expected_protocols_threshold)) == []


def test_get_model_name_for_final(db):
    db.iteration_tag = ""
    assert db.get_model_name_for_final(seed=42) == "cADpyr_L5TPC__42"
    db.iteration_tag = None
    assert db.get_model_name_for_final(seed=42) == "cADpyr_L5TPC__42"
    db.iteration_tag = "hash"
    assert db.get_model_name_for_final(seed=42) == "cADpyr_L5TPC__hash__42"
