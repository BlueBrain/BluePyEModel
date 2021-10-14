"""Model Configurator"""
import logging

import icselector

from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.emodel_pipeline.utils import yesno
from bluepyemodel.model_configuration.neuron_model_configuration import NeuronModelConfiguration

logger = logging.getLogger(__name__)


class ModelConfigurator:
    """Handles the loading and saving and modification of configurations"""

    def __init__(self, access_point, configuration=None):

        self.configuration = configuration
        self.access_point = access_point

    def new_configuration(self, configuration_name, use_gene_data=False):
        """Create a new configuration"""

        if self.configuration is not None:
            self.delete_configuration()

        if use_gene_data:
            self.get_gene_based_configuration(configuration_name=configuration_name)
        else:
            self.configuration = NeuronModelConfiguration(configuration_name=configuration_name)

    def load_configuration(self, name):
        """Load a previously registered configuration"""

        if isinstance(self.access_point, LocalAccessPoint):
            raise Exception("Loading configuration is not yet implemented for local access point")

        self.access_point.get_model_configuration(name)

    def save_configuration(self, path=None):
        """Save the created configuration"""

        if self.configuration:
            self.access_point.store_model_configuration(self.configuration, path)

    def delete_configuration(self):
        """Delete the current configuration"""

        if self.configuration:

            if yesno("Save current configuration ?"):
                self.save_configuration()

            self.configuration = None

    def get_gene_based_parameters(self):
        """Get the gene mapping from Nexus and retrieve the matching parameters and mechanisms
        from the ion channel selector"""

        if not self.access_point.pipeline_settings.name_gene_map:
            logger.warning(
                "No gene mapping name informed. Only parameters registered by the user"
                " will be used."
            )
            return

        _, gene_map_path = self.access_point.load_channel_gene_expression(
            self.access_point.pipeline_settings.name_gene_map
        )

        ic_map_path = self.access_point.load_ic_map()

        selector = icselector.ICSelector(ic_map_path, gene_map_path)
        mechanisms = selector.get_nexus_resources([self.access_point.ttype])
        suffix = {m["name"]: m["name"] for m in mechanisms}
        parameters, mechanisms, distributions = selector.get_cell_config(suffix)

        return parameters, mechanisms, distributions

    def get_gene_based_configuration(self, configuration_name):

        self.configuration = NeuronModelConfiguration(configuration_name=configuration_name)

        selector_params, selector_mechs, selector_distrs = self.get_gene_based_parameters()

        for d in selector_distrs:
            if d["name"] in ["uniform", "constant"]:
                continue
            self.configuration.add_distribution(
                d["name"], d["function"], d.get("parameters", None), d.get("soma_ref_location", 0.5)
            )

        for p in selector_params:
            self.configuration.add_parameter(
                p["name"],
                locations=p["location"],
                value=p["value"],
                mechanism=p.get("mechanism", "global"),
            )

        for m in selector_mechs:
            self.configuration.add_mechanism(
                m["name"],
                locations=m["location"],
                stochastic=m.get("stochastic", None),
            )
