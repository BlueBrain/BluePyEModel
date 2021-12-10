import pytest
from pathlib import Path

from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
from bluepyemodel.access_point.local import LocalAccessPoint

TEST_ROOT = Path(__file__).parents[1]
DATA = TEST_ROOT / "test_data"


@pytest.fixture
def api_config():
    return {
        "emodel": "cADpyr_L5TPC",
        "emodel_dir": DATA,
    }


@pytest.fixture
def pipeline():

    pipe = EModel_pipeline(
        emodel="cADpyr_L5TPC",
        data_access_point='local',
        recipes_path=DATA / "config/recipes.json",
        ttype="test",
        species='mouse',
        brain_region='SSCX'
    )

    pipe.access_point.emodel_dir = DATA

    return pipe


def test_init(pipeline):
    assert isinstance(pipeline.access_point, LocalAccessPoint)
