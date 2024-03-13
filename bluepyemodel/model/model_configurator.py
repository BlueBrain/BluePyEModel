"""Model Configurator"""

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

from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.model.neuron_model_configuration import NeuronModelConfiguration
from bluepyemodel.tools.utils import yesno

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

    def new_configuration(self, use_gene_data=False):
        """Create a new configuration"""

        if self.configuration is not None:
            self.delete_configuration()

        if use_gene_data:
            self.get_gene_based_configuration()
        else:
            self.configuration = NeuronModelConfiguration(
                distributions=self.access_point.get_distributions(),
                available_mechanisms=self.access_point.get_available_mechanisms(),
                available_morphologies=self.access_point.get_available_morphologies(),
            )

    def load_configuration(self):
        """Load a previously registered configuration"""

        if isinstance(self.access_point, LocalAccessPoint):
            raise NotImplementedError(
                "Loading configuration is not yet implemented for local access point"
            )

        self.configuration = self.access_point.get_model_configuration()

    def save_configuration(self, path=None):
        """Save the configuration. The saving medium depends on the access point."""

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

        try:
            import icselector
        except ImportError as exc:
            raise ImportError(
                "The internal icselector package is required to use gene based configuration."
            ) from exc

        if not self.access_point.pipeline_settings.name_gene_map:
            logger.warning(
                "No gene mapping name informed. Only parameters registered by the user"
                " will be used."
            )
            return [], [], [], []

        _, gene_map_path = self.access_point.load_channel_gene_expression(
            self.access_point.pipeline_settings.name_gene_map
        )

        ic_map_path = self.access_point.load_ic_map()

        selector = icselector.ICSelector(ic_map_path, gene_map_path)
        parameters, mechanisms, distributions, nexus_keys = selector.get_cell_config_from_ttype(
            self.access_point.emodel_metadata.ttype
        )

        return parameters, mechanisms, distributions, nexus_keys

    def get_gene_based_configuration(self):
        """Overwrite the currently loaded configuration with a new configuration initiated from
        gene data."""

        self.configuration = NeuronModelConfiguration(
            distributions=self.access_point.get_distributions(),
            available_mechanisms=self.access_point.get_available_mechanisms(),
            available_morphologies=self.access_point.get_available_morphologies(),
        )

        params, mechs, distributions, nexus_keys = self.get_gene_based_parameters()

        for d in distributions:
            if d["name"] in ["uniform", "constant"]:
                continue
            function = d["function"] if "function" in d else d["fun"]
            self.configuration.add_distribution(
                d["name"], function, d.get("parameters", None), d.get("soma_ref_location", 0.5)
            )

        for p in params:
            self.configuration.add_parameter(
                p["name"],
                locations=p["location"],
                value=p["value"],
                mechanism=p.get("mechanism", "global"),
            )

        for m in mechs:
            version = None
            for k in nexus_keys:
                if k["name"] == m["name"]:
                    version = k.get("modelid") if "modelid" in k else k.get("modelId", None)

            temp_entry = m.get("temperature", None)
            if isinstance(temp_entry, dict) and "value" in temp_entry:
                temperature = temp_entry.get("value")
            else:
                temperature = temp_entry

            self.configuration.add_mechanism(
                m["name"],
                locations=m["location"],
                stochastic=m.get("stochastic", None),
                version=version,
                temperature=temperature,
                ljp_corrected=(
                    m.get("ljp_corrected")
                    if "ljp_corrected" in m
                    else m.get("isLjpCorrected", None)
                ),
            )
