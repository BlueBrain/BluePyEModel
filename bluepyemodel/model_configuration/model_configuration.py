"""Model configuration related functions"""
import logging

from bluepyemodel.model_configuration.model_configurator import ModelConfigurator

logger = logging.getLogger(__name__)


def configure_model(
    access_point,
    morphology_name,
    emodel=None,
    ttype=None,
    use_gene_data=True,
):
    """Creates a model configuration: parameters, distributions, mechanisms, ...

    Args:
        access_point (DataAccessPoint): object which contains API to access emodel data
        emodel (str): name of the emodel.
        ttype (str): name of the transcriptomic type.
        use_gene_data (bool): should the configuration be initialized using gene data
    """

    if access_point.pipeline_settings.model_configuration_name:
        configuration_name = access_point.pipeline_settings.model_configuration_name
    else:
        configuration_name = f"{emodel}_{ttype}"

    configurator = ModelConfigurator(access_point=access_point)
    configurator.new_configuration(
        configuration_name=configuration_name, use_gene_data=use_gene_data
    )

    if morphology_name:
        configurator.configuration.select_morphology(morphology_name)

    configurator.save_configuration()

    return configurator.configuration
