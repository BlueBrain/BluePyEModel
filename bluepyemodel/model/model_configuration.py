"""Model configuration related functions"""

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
        morphology_name,
        morphology_path=morphology_path,
        morphology_format=morphology_format,
    )

    configurator.save_configuration()

    return configurator.configuration
