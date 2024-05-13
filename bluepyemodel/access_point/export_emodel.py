"""Export the emodels to Nexus"""

import logging
import time

from bluepyemodel.export_emodel.export_emodel import select_emodels

# pylint: disable=too-many-locals

logger = logging.getLogger(__name__)


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
