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

import json
from pathlib import Path

from dictdiffer import diff

from tests.utils import DATA


def test_get_morphologies(db_from_nexus):
    morphology = db_from_nexus.get_morphologies()
    assert morphology["name"] == "C060114A5"
    assert Path(morphology["path"]).name == "C060114A5.asc"


def test_get_available_morphologies(db_from_nexus):
    names = db_from_nexus.get_available_morphologies()
    assert len(names) == 1
    assert list(names)[0] == "C060114A5"


def test_get_recipes(db_from_nexus, emodel_dir):
    recipes = db_from_nexus.get_recipes()
    expected_recipes = json.load(open(db_from_nexus.recipes_path, "r"))["L5PC"]
    assert list(diff(recipes, expected_recipes)) == []


def test_get_model_configuration(db_from_nexus):
    configuration = db_from_nexus.get_model_configuration()

    expected_parameters = json.load(open(DATA / "test_parameters.json", "r"))

    for p in configuration.parameters:
        assert p.location in expected_parameters["parameters"]
        for ep in expected_parameters["parameters"][p.location]:
            if ep["name"] == p.name and ep["val"] == p.value:
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


def test_get_final_content(db_from_nexus):
    final = db_from_nexus.get_final_content()
    assert "L5PC" in final
    assert "parameters" in final["L5PC"]


def test_load_pipeline_settings(db_from_nexus):
    assert (
        db_from_nexus.pipeline_settings.path_extract_config
        == "config/extract_config/L5PC_config.json"
    )
    assert db_from_nexus.pipeline_settings.validation_protocols == ["sAHP_220"]


def test_get_model_name_for_final(db_from_nexus):
    db_from_nexus.emodel_metadata.iteration = ""
    assert db_from_nexus.get_model_name_for_final(seed=42) == "L5PC__42"
    db_from_nexus.emodel_metadata.iteration = None
    assert db_from_nexus.get_model_name_for_final(seed=42) == "L5PC__42"
    db_from_nexus.emodel_metadata.iteration = "hash"
    assert db_from_nexus.get_model_name_for_final(seed=42) == "L5PC__hash__42"


def test_get_ion_currents_concentrations(db_from_nexus):
    expected_ion_currents = {
        "ica_Ca_HVA2",
        "ica_Ca_LVAst",
        "ik_K_Pst",
        "ik_K_Tst",
        "ina_NaTg",
        "ina_Nap_Et2",
        "ik_SK_E2",
        "ik_SKv3_1",
        "ihcn_Ih",
        "i_pas",
    }
    expected_ionic_concentrations = {
        "cai",
        "ki",
        "nai",
    }
    ion_currents, ionic_concentrations = db_from_nexus.get_ion_currents_concentrations()
    assert set(ion_currents) == expected_ion_currents
    assert set(ionic_concentrations) == expected_ionic_concentrations
