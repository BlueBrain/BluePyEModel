"""
Copyright 2024, EPFL/Blue Brain Project

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

import os
import shutil

import pytest

from bluepyemodel.access_point import get_access_point
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from tests.utils import DATA, cwd


@pytest.fixture(scope="session")
def nrnivmodl(tmp_path_factory):
    """Compile the mechanisms only once per session."""
    path = tmp_path_factory.mktemp("nrnivmodl_dir")
    with cwd(path):
        os.popen(f"nrnivmodl {DATA}/mechanisms").read()
    return path


@pytest.fixture
def workspace(tmp_path):
    """Change the working directory to tmp_path.

    Any local directory (for example .tmp and x86_64) may be created here.
    """
    with cwd(tmp_path):
        yield tmp_path


@pytest.fixture
def emodel_dir(workspace, nrnivmodl):
    """Copy the required files to workspace/emodel."""
    dirs = ["config", "mechanisms", "morphology", "ephys_data"]
    files = ["final.json"]
    dst = workspace / "emodel"
    for name in dirs:
        shutil.copytree(DATA / name, dst / name)
    for name in files:
        shutil.copyfile(DATA / name, dst / name)
    shutil.copytree(nrnivmodl / "x86_64", workspace / "x86_64")
    yield dst


@pytest.fixture
def api_config(emodel_dir):
    return {
        "emodel": "cADpyr_L5TPC",
        "emodel_dir": emodel_dir,
        "recipes_path": emodel_dir / "config/recipes.json",
    }


@pytest.fixture
def db(api_config):
    return get_access_point("local", **api_config)


@pytest.fixture
def db_restart(emodel_dir):
    return get_access_point(
        "local",
        emodel="cADpyr_L5TPC",
        emodel_dir=emodel_dir,
        recipes_path=emodel_dir / "config/recipes_restart.json",
    )


@pytest.fixture
def db_from_nexus(emodel_dir):
    return get_access_point(
        "local",
        emodel="L5PC",
        emodel_dir=emodel_dir,
        recipes_path=emodel_dir / "config/recipes_nexus.json",
    )
