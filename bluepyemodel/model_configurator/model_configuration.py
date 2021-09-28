"""Model configuration related functions"""
import logging

import icselector

from bluepyemodel.model_configurator.neuron_model_configuration import NeuronModelConfiguration

logger = logging.getLogger(__name__)


def get_gene_based_parameters(ttype, access_point):
    """Get the gene mapping from Nexus and retrieve the matching parameters and mechanisms
    from the ion channel selector"""

    if not access_point.pipeline_settings.name_gene_map:
        logger.warning(
            "No gene mapping name informed. Only parameters registered by the user" " will be used."
        )
        return

    _, gene_map_path = access_point.load_channel_gene_expression(
        access_point.pipeline_settings.name_gene_map
    )

    ic_map_path = access_point.load_ic_map()

    selector = icselector.ICSelector(ic_map_path, gene_map_path)
    mechanisms = selector.get_nexus_resources([ttype])
    suffix = {m["name"]: m["name"] for m in mechanisms}
    parameters, mechanisms, distributions = selector.get_cell_config(suffix)

    return parameters, mechanisms, distributions


def configure_model(
    access_point,
    emodel,
    ttype,
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

    model_configuration = NeuronModelConfiguration(configuration_name=configuration_name)

    if use_gene_data:

        selecto_params, selector_mechs, selector_distrs = get_gene_based_parameters(
            ttype, access_point
        )

        for d in selector_distrs:
            if d["name"] in ["uniform", "constant"]:
                continue
            model_configuration.add_distribution(
                d["name"], d["function"], d.get("parameters", None), d.get("soma_ref_location", 0.5)
            )

        for p in selecto_params:
            model_configuration.add_parameter(
                p["name"],
                locations=p["location"],
                value=p["value"],
                mechanism=p.get("mechanism", "global"),
            )

        for m in selector_mechs:
            model_configuration.add_mechanism(
                m["name"],
                locations=m["location"],
                stochastic=p.get("stochastic", None),
            )

    # TODO: add user based modifications / GUI

    access_point.store_model_configuration(model_configuration)

    return model_configuration
