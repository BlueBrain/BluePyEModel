"""Export the emodels in the SONATA format"""

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

import logging
import os
import pathlib
import shutil
import time

import h5py

from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point
from bluepyemodel.export_emodel.utils import get_hoc_file_path
from bluepyemodel.export_emodel.utils import get_output_path
from bluepyemodel.export_emodel.utils import select_emodels

# pylint: disable=too-many-locals

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
    contain a sonata nodes.h5 file, the morphology of the model and a hoc version of the model.
    """

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


def export_emodels_nexus(
    local_access_point,
    nexus_organisation,
    nexus_project,
    nexus_endpoint="https://bbp.epfl.ch/nexus/v1",
    forge_path=None,
    forge_ontology_path=None,
    access_token=None,
    only_validated=False,
    only_best=True,
    seeds=None,
    description=None,
    sleep_time=10,
    sonata=True,
):
    """Transfer e-models from the LocalAccessPoint to a Nexus project

    Args:
        local_access_point (LocalAccessPoint): The local access point containing the e-models.
        nexus_organisation (str): The Nexus organisation to which the e-models will be transferred.
        nexus_project (str): The Nexus project to which the e-models will be transferred.
        nexus_endpoint (str, optional): The Nexus endpoint.
            Defaults to "https://bbp.epfl.ch/nexus/v1".
        forge_path (str, optional): The path to the forge.
        forge_ontology_path (str, optional): The path to the forge ontology.
        access_token (str, optional): The access token for Nexus.
        only_validated (bool, optional): If True, only validated e-models will be transferred.
        only_best (bool, optional): If True, only the best e-models will be transferred.
        seeds (list, optional): The chosen seeds to export.
        description (str, optional): Optional description to add to the resources in Nexus.
        sleep_time (int, optional):  time to wait between two Nexus requests
            (in case of slow indexing).
        sonata (bool, optional): Determines the format for registering e-models.
            If True (default), uses Sonata hoc format. Otherwise, uses NEURON hoc format.

    Returns:
        None
    """

    from bluepyemodel.access_point.nexus import NexusAccessPoint

    emodels = local_access_point.get_emodels()
    emodels = select_emodels(
        local_access_point.emodel_metadata.emodel,
        emodels,
        only_validated=only_validated,
        only_best=only_best,
        seeds=seeds,
    )
    if not emodels:
        return

    metadata = vars(local_access_point.emodel_metadata)
    iteration = metadata.pop("iteration")
    metadata.pop("allen_notation")
    nexus_access_point = NexusAccessPoint(
        **metadata,
        iteration_tag=iteration,
        project=nexus_project,
        organisation=nexus_organisation,
        endpoint=nexus_endpoint,
        access_token=access_token,
        forge_path=forge_path,
        forge_ontology_path=forge_ontology_path,
        sleep_time=sleep_time,
    )

    pipeline_settings = local_access_point.pipeline_settings
    fitness_configuration = local_access_point.get_fitness_calculator_configuration()
    model_configuration = local_access_point.get_model_configuration()
    targets_configuration = local_access_point.get_targets_configuration()

    # Register the resources
    logger.info("Exporting the emodel %s to Nexus...", local_access_point.emodel_metadata.emodel)
    logger.info("Registering EModelPipelineSettings...")
    nexus_access_point.store_pipeline_settings(pipeline_settings)

    logger.info("Registering ExtractionTargetsConfiguration...")
    # Set local filepath to None to avoid discrepancies between local and Nexus paths
    for file in targets_configuration.files:
        file.filepath = None
    nexus_access_point.store_targets_configuration(targets_configuration)

    logger.info("Registering EModelConfiguration...")
    # Remove unused local data from the model configuration before uploading to Nexus
    model_configuration.morphology.path = None
    nexus_access_point.store_model_configuration(model_configuration)

    logger.info("Registering EModelWorkflow...")
    filters = {"type": "EModelWorkflow", "eModel": metadata["emodel"], "iteration": iteration}
    filters_legacy = {
        "type": "EModelWorkflow",
        "emodel": metadata["emodel"],
        "iteration": iteration,
    }
    nexus_access_point.access_point.deprecate(filters, filters_legacy)
    time.sleep(sleep_time)
    emw = nexus_access_point.create_emodel_workflow(state="done")
    nexus_access_point.store_or_update_emodel_workflow(emw)

    logger.info("Registering FitnessCalculatorConfiguration...")
    time.sleep(sleep_time)
    nexus_access_point.store_fitness_calculator_configuration(fitness_configuration)

    for mo in emodels:
        time.sleep(sleep_time)
        mo.emodel_metadata.allen_notation = nexus_access_point.emodel_metadata.allen_notation
        mo.copy_pdf_dependencies_to_new_path(seed=mo.seed)
        logger.info("Registering EModel %s...", mo.emodel_metadata.emodel)
        nexus_access_point.store_emodel(mo, description=description)

    time.sleep(sleep_time)
    if sonata:
        logger.info(
            "Registering EModelScript (in sonata hoc format with threshold_current and "
            "holding_current in node.h5 file) for circuit building using neurodamus..."
        )
        nexus_access_point.store_emodels_sonata(
            only_best=only_best,
            only_validated=only_validated,
            seeds=seeds,
            description=description,
        )
    else:
        logger.info("Registering EModelScript (in hoc format to run e-model using NEURON)...")
        nexus_access_point.store_emodels_hoc(
            only_best=only_best,
            only_validated=only_validated,
            seeds=seeds,
            description=description,
        )

    logger.info(
        "Exporting the emodel %s to Nexus done.",
        local_access_point.emodel_metadata.emodel,
    )
