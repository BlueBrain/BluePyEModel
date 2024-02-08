"""Export the emodels in the SONATA format"""

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

import logging
import os
import pathlib
import shutil

import h5py

from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.export_emodel.utils import get_hoc_file_path
from bluepyemodel.export_emodel.utils import get_output_path
from bluepyemodel.export_emodel.utils import select_emodels

logger = logging.getLogger(__name__)


def _write_node_file(emodel, model_template_path, node_file_path, morphology_path=None):
    """Creates a nodes.h5 file in the SONATA format. It contains the information
    needed to run the model. See https://bbpteam.epfl.ch/documentation/projects/
    circuit-documentation/latest/sonata_tech.html"""

    with h5py.File(node_file_path, "w") as f:
        population_name = f"{emodel.emodel_metadata.brain_region}_neurons"
        population = f.create_group(f"/nodes/{population_name}/0")

        population.create_dataset("x", (1,), dtype="f")
        population.create_dataset("y", (1,), dtype="f")
        population.create_dataset("z", (1,), dtype="f")

        population.create_dataset("orientation_w", (1,), dtype="f")
        population.create_dataset("orientation_x", (1,), dtype="f")
        population.create_dataset("orientation_y", (1,), dtype="f")
        population.create_dataset("orientation_z", (1,), dtype="f")

        population["model_template"] = str(model_template_path)
        population["model_type"] = "biophysical"

        if morphology_path:
            population["morphology"] = pathlib.Path(morphology_path).stem

        for k in ["etype", "mtype", "synapse_class"]:
            v = getattr(emodel.emodel_metadata, k)
            if v is not None:
                population[k] = v

        if emodel.emodel_metadata.brain_region is not None:
            population["region"] = emodel.emodel_metadata.brain_region

        threshold_keys = ("bpo_holding_current", "bpo_threshold_current")
        if all(key in emodel.responses for key in threshold_keys):
            dynamics_params = population.create_group("dynamics_params")
            dynamics_params["holding_current"] = emodel.responses["bpo_holding_current"]
            dynamics_params["threshold_current"] = emodel.responses["bpo_threshold_current"]


def _write_hoc_file(
    cell_model,
    emodel,
    hoc_file_path,
    template="cell_template_neurodamus_sbo.jinja2",
    new_emodel_name=None,
):
    """Creates a hoc file containing the emodel and its morphology.
    WARNING: this assumes that any morphology modifier has been informed as both
    a python method and a hoc method"""

    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))

    if new_emodel_name is not None:
        cell_model.name = new_emodel_name

    hoc_content = cell_model.create_hoc(
        param_values=emodel.parameters,
        template=template,
        template_dir=template_dir,
    )

    with open(hoc_file_path, "w") as f:
        f.writelines(hoc_content)


def _export_model_sonata(cell_model, emodel, output_dir=None, new_emodel_name=None):
    """Creates the directory and files required for an emodel to be used in circuit building"""

    if not emodel.passed_validation:
        logger.warning("Exporting a model that did not pass validation.")

    if new_emodel_name is not None:
        emodel.emodel_metadata.emodel = new_emodel_name

    output_path = get_output_path(emodel, output_dir, output_base_dir="export_emodels_sonata")
    hoc_file_path = get_hoc_file_path(output_path)
    node_file_path = str(output_path / "nodes.h5")
    morphology_path = str(output_path / pathlib.Path(cell_model.morphology.morphology_path).name)

    # Copy the morphology
    shutil.copyfile(cell_model.morphology.morphology_path, morphology_path)

    # Exports the BluePyOpt cell model as a hoc file
    _write_hoc_file(
        cell_model,
        emodel,
        hoc_file_path,
        template="cell_template_neurodamus_sbo.jinja2",
        new_emodel_name=new_emodel_name,
    )

    # Create the SONATA node file
    _write_node_file(
        emodel,
        model_template_path=hoc_file_path,
        node_file_path=node_file_path,
        morphology_path=morphology_path,
    )


def export_emodels_sonata(
    access_point,
    only_validated=False,
    only_best=True,
    seeds=None,
    map_function=map,
    new_emodel_name=None,
    new_metadata=None,
):
    """Export a set of emodels to a set of folder named after them. Each folder will
    contain a sonata nodes.h5 file, the morphology of the model and a hoc version of the model."""

    cell_evaluator = get_evaluator_from_access_point(
        access_point, include_validation_protocols=True
    )

    emodels = compute_responses(
        access_point,
        cell_evaluator,
        map_function,
        seeds=seeds,
        preselect_for_validation=False,
        store_responses=False,
    )

    emodels = select_emodels(
        access_point.emodel_metadata.emodel,
        emodels,
        only_validated=only_validated,
        only_best=only_best,
        seeds=seeds,
    )
    if not emodels:
        logger.warning(
            "No emodels were selected in export_emodels_sonata. Stopping sonata export here."
        )
        return

    cell_model = cell_evaluator.cell_model

    for mo in emodels:
        if new_metadata:
            mo.emodel_metadata = new_metadata
        if not cell_model.morphology.morph_modifiers:  # Turn [] into None
            cell_model.morphology.morph_modifiers = None
        _export_model_sonata(cell_model, mo, output_dir=None, new_emodel_name=new_emodel_name)


def _export_emodel_hoc(cell_model, mo, output_dir=None, new_emodel_name=None):
    if not mo.passed_validation:
        logger.warning("Exporting a model that did not pass validation.")

    if new_emodel_name is not None:
        mo.emodel_metadata.emodel = new_emodel_name

    output_path = get_output_path(mo, output_dir, output_base_dir="export_emodels_hoc")
    hoc_file_path = get_hoc_file_path(output_path)
    morphology_path = str(output_path / pathlib.Path(cell_model.morphology.morphology_path).name)

    # Copy the morphology
    shutil.copyfile(cell_model.morphology.morphology_path, morphology_path)

    # Exports the BluePyOpt cell model as a hoc file
    _write_hoc_file(
        cell_model,
        mo,
        hoc_file_path,
        template="cell_template.jinja2",
        new_emodel_name=new_emodel_name,
    )


def export_emodels_hoc(
    access_point,
    only_validated=False,
    only_best=True,
    seeds=None,
    new_emodel_name=None,
    new_metadata=None,
):
    """Export a set of emodels to a set of folder named after them. Each folder will contain a hoc
    version of the model."""

    cell_evaluator = get_evaluator_from_access_point(
        access_point, include_validation_protocols=True
    )

    emodels = access_point.get_emodels()

    emodels = select_emodels(
        access_point.emodel_metadata.emodel,
        emodels,
        only_validated=only_validated,
        only_best=only_best,
        seeds=seeds,
        iteration=access_point.emodel_metadata.iteration,
    )
    if not emodels:
        logger.warning("No emodels were selected in export_emodels_hoc. Stopping hoc export here.")
        return

    cell_model = cell_evaluator.cell_model

    for mo in emodels:
        if new_metadata:
            mo.emodel_metadata = new_metadata
        if not cell_model.morphology.morph_modifiers:  # Turn [] into None
            cell_model.morphology.morph_modifiers = None
        _export_emodel_hoc(cell_model, mo, output_dir=None, new_emodel_name=new_emodel_name)
