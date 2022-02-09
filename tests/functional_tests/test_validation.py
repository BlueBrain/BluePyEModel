from pathlib import Path
import pytest
import logging

from bluepyemodel.access_point import get_access_point
from bluepyemodel.validation.validation import validate, define_validation_function

TEST_ROOT = Path(__file__).parents[1]
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


def always_true_validation(model, threshold=5.0, validation_protocols_only=False):
    return True


def test_define_validation_function(db):

    model = {
        "scores": {"a": 0.0, "b": 4.9, "c": 0.5, "d": 9.9},
        "scores_validation": {"c": 0.5, "d": 9.9},
    }

    db.pipeline_settings.validation_function = always_true_validation

    validation_function = define_validation_function(db)

    validated = bool(
        validation_function(
            model,
            db.pipeline_settings.validation_threshold,
            False,
        )
    )

    assert validated


def test_validation(db):

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    
    db.get_mechanisms_directory = lambda: None
    emodels = validate(
        access_point=db,
        mapper=map,
    )

    assert emodels[0].passed_validation
