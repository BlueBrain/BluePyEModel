"""Model Configurator"""
import logging

import icselector

from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.emodel_pipeline.utils import yesno
from bluepyemodel.model.neuron_model_configuration import NeuronModelConfiguration

logger = logging.getLogger(__name__)


class ModelConfigurator:
    """Handles the loading, saving and modification of a model configuration"""

    def __init__(self, access_point, configuration=None):
        """Creates a model configuration, which includes the model parameters, distributions,
        mechanisms and a morphology.

        Args:
            access_point (DataAccessPoint): access point to the emodel data.
            configuration (NeuronModelConfiguration): a pre-existing configuration.
        """

        self.access_point = access_point
        self.configuration = configuration

    def new_configuration(self, configuration_name, use_gene_data=False):
        """Create a new configuration"""

        if self.configuration is not None:
            self.delete_configuration()

        if use_gene_data:
            self.get_gene_based_configuration(configuration_name=configuration_name)
        else:
            self.configuration = NeuronModelConfiguration(
                configuration_name=configuration_name,
                available_mechanisms=self.access_point.get_available_mechanisms(),
                available_morphologies=self.access_point.get_available_morphologies(),
            )

    def load_configuration(self, name):
        """Load a previously registered configuration"""

        if isinstance(self.access_point, LocalAccessPoint):
            raise Exception("Loading configuration is not yet implemented for local access point")

        self.access_point.get_model_configuration(name)

    def save_configuration(self, path=None):
        """Save the configuration. The saving medium depends of the access point."""

        if self.configuration:
            self.access_point.store_model_configuration(self.configuration, path)

    def delete_configuration(self):
        """Delete the current configuration. Warning: it does not delete the file or resource of
        the configuration."""

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
            return [], [], []

        _, gene_map_path = self.access_point.load_channel_gene_expression(
            self.access_point.pipeline_settings.name_gene_map
        )

        ic_map_path = self.access_point.load_ic_map()

        selector = icselector.ICSelector(ic_map_path, gene_map_path)
        parameters, mechanisms, distributions, _ = selector.get_cell_config_from_ttype(
            self.access_point.ttype
        )

        return parameters, mechanisms, distributions

    def get_gene_based_configuration(self, configuration_name):
        """Overwrite the currently loaded configuration with a new configuration initiated from
        gene data."""

        self.configuration = NeuronModelConfiguration(
            configuration_name=configuration_name,
            available_mechanisms=self.access_point.get_available_mechanisms(),
            available_morphologies=self.access_point.get_available_morphologies(),
        )

        selector_params, selector_mechs, selector_distrs = self.get_gene_based_parameters()

        for d in selector_distrs:
            if d["name"] in ["uniform", "constant"]:
                continue
            function = d["function"] if "function" in d else d["fun"]
            self.configuration.add_distribution(
                d["name"], function, d.get("parameters", None), d.get("soma_ref_location", 0.5)
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