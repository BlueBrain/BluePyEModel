"""Export the emodels in the SONATA format"""
import logging
import os
import pathlib
import shutil

import h5py

from bluepyemodel.evaluation.evaluation import compute_responses
from bluepyemodel.evaluation.evaluation import get_evaluator_from_access_point

logger = logging.getLogger(__name__)


def write_node_file(emodel, model_template_path, node_file_path, morphology_path=None):
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

        for k in ["layer", "morph_class", "etype", "mtype", "synapse_class"]:
            v = getattr(emodel.emodel_metadata, k)
            if v is not None:
                population[k] = v

        if emodel.emodel_metadata.brain_region is not None:
            population["region"] = emodel.emodel_metadata.brain_region

        if "bpo_holding_current" in emodel.responses:
            dynamics_params = population.create_group("dynamics_params")
            dynamics_params["holding_current"] = emodel.responses["bpo_holding_current"]
            dynamics_params["threshold_current"] = emodel.responses["bpo_threshold_current"]


def write_hoc_file(cell_model, emodel, hoc_file_path):
    """Creates a hoc file containing the emodel and its morphology.
    WARNING: this assumes that any morphology modifier has been informed as both
    a python method and a hoc method"""

    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))

    hoc_content = cell_model.create_hoc(
        param_values=emodel.parameters,
        template="cell_template_neurodamus.jinja2",
        template_dir=template_dir,
    )

    with open(hoc_file_path, "w") as f:
        f.writelines(hoc_content)


def export_model_sonata(cell_model, emodel, output_dir=None):
    """Creates the directory and files required for an emodel to be used in circuit building"""

    if not emodel.passed_validation:
        logger.warning("Exporting a model that did not pass validation.")

    if output_dir is None:
        output_dir = f"./export_emodels/{emodel.emodel_metadata.as_string(seed=emodel.seed)}/"
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hoc_file_path = str(output_path / "model.hoc")
    node_file_path = str(output_path / "nodes.h5")
    morphology_path = str(output_path / pathlib.Path(cell_model.morphology.morphology_path).name)

    # Copy the morphology
    shutil.copyfile(cell_model.morphology.morphology_path, morphology_path)

    # Exports the BluePyOpt cell model as a hoc file
    write_hoc_file(cell_model, emodel, hoc_file_path)

    # Create the SONATA node file
    write_node_file(
        emodel,
        model_template_path=hoc_file_path,
        node_file_path=node_file_path,
        morphology_path=morphology_path,
    )


def export_emodels(access_point, only_validated=False, seeds=None, map_function=map):
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

    if only_validated:
        emodels = [model for model in emodels if model.passed_validation]

    if emodels:

        logger.info("In export_emodels, %s emodels found to export.", len(emodels))
        cell_model = cell_evaluator.cell_model

        for mo in emodels:
            if not cell_model.morphology.morph_modifiers:  # Turn [] into None
                cell_model.morphology.morph_modifiers = None
            export_model_sonata(cell_model, mo, output_dir=None)

    else:
        logger.warning("In export_emodels, no emodel for %s", access_point.emodel_metadata.emodel)
