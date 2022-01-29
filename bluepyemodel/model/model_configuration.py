"""Model configuration related functions"""
import logging

from bluepyemodel.model.model_configurator import ModelConfigurator

logger = logging.getLogger(__name__)


def configure_model(
    access_point,
    morphology_name,
    morphology_path=None,
    morphology_format=None,
    use_gene_data=True,
):
    """Task of creating and saving a neuron model configuration.

    Args:
        access_point (DataAccessPoint): access point to the emodel data
        morphology_name (str): name of the morphology on which to build the neuron model. This
            morphology has to be available in the directory "./morphology" if using the local
            access point or on Nexus if using the Nexus access point.
        morphology_path (str): path to the morphology file
        morphology_format (str): format of the morphology, can be 'asc' or 'swc'. Optional if
            morphology_path was provided.
        use_gene_data (bool): should the configuration be initialized using gene data. If False,
            the configuration will be empty.
    """

    configurator = ModelConfigurator(access_point=access_point)
    configurator.new_configuration(use_gene_data=use_gene_data)

    configurator.configuration.select_morphology(
        morphology_name, morphology_path=morphology_path, morphology_format=morphology_format
    )

    configurator.save_configuration()

    return configurator.configuration
