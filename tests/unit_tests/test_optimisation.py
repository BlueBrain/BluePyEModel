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

from bluepyemodel.tools.utils import get_checkpoint_path
from bluepyemodel.tools.utils import get_legacy_checkpoint_path
from bluepyemodel.tools.utils import get_seed_from_checkpoint_path
from bluepyemodel.emodel_pipeline.emodel_metadata import EModelMetadata


def checkpoint_check(dir, fname, metadata, inner_dir):
    f = dir / fname
    f.touch()
    assert str(get_checkpoint_path(metadata, seed=0)) == "/".join((".", inner_dir, fname))
    f.unlink()

def test_get_checkpoint_path(workspace):
    metadata = EModelMetadata(
        emodel="L5PC",
        mtype="L5TPC:A",
        ttype="t type",
        iteration_tag="test",
        brain_region="somatosensory cortex",
        allen_notation="SSCX",
    )
    path = get_checkpoint_path(metadata, seed=0)
    fname = "emodel=L5PC__ttype=t_type__mtype=L5TPC_A__brain_region=SSCX__iteration=test__seed=0.pkl"
    assert (
        str(path) == f"./checkpoints/L5PC/test/{fname}"
    )
    path = get_legacy_checkpoint_path(path)
    assert str(path) == f"./checkpoints/{fname}"

    # test also legacy formats
    inner_dir = "checkpoints/L5PC/test"
    dir = workspace / inner_dir
    dir.mkdir(parents=True)
    fname = "emodel=L5PC__ttype=t type__mtype=L5TPC:A__brain_region=somatosensory cortex__iteration=test__seed=0.pkl"
    checkpoint_check(dir, fname, metadata, inner_dir)
    fname = "emodel=L5PC__ttype=t type__mtype=L5TPC:A__brain_region=SSCX__iteration=test__seed=0.pkl"
    checkpoint_check(dir, fname, metadata, inner_dir)
    fname = "emodel=L5PC__ttype=t type__mtype=L5TPC_A__brain_region=SSCX__iteration=test__seed=0.pkl"
    checkpoint_check(dir, fname, metadata, inner_dir)


def test_get_seed_from_checkpoint_path():
    seed = get_seed_from_checkpoint_path(
        "./checkpoints/L5PC/test/emodel=L5PC__seed=0__iteration=test__ttype=t_type.pkl"
    )
    assert seed == 0
