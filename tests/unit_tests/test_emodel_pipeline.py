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

import pytest

from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.emodel_pipeline.emodel_pipeline import EModel_pipeline
from tests.utils import DATA


@pytest.fixture
def pipeline():

    pipe = EModel_pipeline(
        emodel="cADpyr_L5TPC",
        recipes_path=DATA / "config/recipes.json",
        ttype="test",
        species='mouse',
        brain_region='SSCX'
    )

    pipe.access_point.emodel_dir = DATA

    return pipe


def test_init(pipeline):
    assert isinstance(pipeline.access_point, LocalAccessPoint)
    with pytest.raises(
        ValueError,
        match="Attempted to set a legacy variable. "
        "This variable should not be modified in new code."
    ):
        _ = EModel_pipeline(
            emodel="cADpyr_L5TPC",
            recipes_path=DATA / "config/recipes.json",
            ttype="test",
            species='mouse',
            brain_region='SSCX',
            data_access_point="nexus"
        )
