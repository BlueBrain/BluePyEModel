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

from bluepyemodel.tools.utils import get_checkpoint_path
from bluepyemodel.tools.utils import get_legacy_checkpoint_path
from bluepyemodel.tools.utils import parse_checkpoint_path
from bluepyemodel.emodel_pipeline.emodel_metadata import EModelMetadata


def test_get_checkpoint_path():
    metadata = EModelMetadata(emodel="L5PC", ttype="t type", iteration_tag="test")
    path = get_checkpoint_path(metadata, seed=0)
    assert str(path) == "./checkpoints/L5PC/test/emodel=L5PC__ttype=t type__iteration=test__seed=0.pkl"
    path = get_legacy_checkpoint_path(path)
    assert str(path) == "./checkpoints/emodel=L5PC__ttype=t type__iteration=test__seed=0.pkl"


def test_parse_checkpoint_path():

    metadata = parse_checkpoint_path(
        "./checkpoints/L5PC/test/emodel=L5PC__seed=0__iteration=test__ttype=t type.pkl"
    )
    for k, v in {
        "emodel": "L5PC",
        "seed": "0",
        "ttype": "t type",
        "iteration": "test"
    }.items():
        assert metadata[k] == v

    metadata = parse_checkpoint_path(
        "./checkpoints/L5PC/test/checkpoint__L5PCpyr_ET1_dend__b6f7190__6.pkl"
    )

    for k, v in {
        "emodel": "L5PCpyr_ET1_dend",
        "seed": "6",
        "ttype": None,
        "iteration": "b6f7190"
    }.items():
        assert metadata[k] == v
