"""EModelWorkflow class"""

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

logger = logging.getLogger(__name__)


class EModelWorkflow:
    """Contains the state of the workflow and the configurations needed for the workflow"""

    def __init__(
        self,
        targets_configuration_id,
        pipeline_settings_id,
        emodel_configuration_id,
        fitness_configuration_id=None,
        emodels=None,
        emodel_scripts_id=None,
        state="not launched",
    ):
        """Init

        Args:
            targets_configuration (str): TargetsConfiguration nexus id
            pipeline_settings (str): EModelPipelineSettings nexus id
            emodel_configuration (str): NeuronModelConfiguration id
            fitness_configuration_id (str): FitnessCalculatorConfiguration id
            emodels (list): list of EModel ids
            state (str): can be "not launched", "running" or "done"
        """
        self.targets_configuration_id = targets_configuration_id
        self.pipeline_settings_id = pipeline_settings_id
        self.emodel_configuration_id = emodel_configuration_id
        self.fitness_configuration_id = fitness_configuration_id
        self.emodels = emodels if emodels else []
        self.emodel_scripts_id = emodel_scripts_id if emodel_scripts_id else []
        self.state = state

    def add_emodel_id(self, emodel_id):
        """Add an emodel id to the list of emodels"""
        self.emodels.append(emodel_id)

    def add_emodel_script_id(self, emodel_script_id):
        """Add an emodel id to the list of emodels"""
        self.emodel_scripts_id.append(emodel_script_id)

    def get_configuration_ids(self):
        """Return all configuration id parameters"""
        ids = (
            self.targets_configuration_id,
            self.pipeline_settings_id,
            self.emodel_configuration_id,
        )
        if self.fitness_configuration_id:
            ids += tuple([self.fitness_configuration_id])

        return ids

    def get_related_nexus_ids(self):
        emodels_ids = [{"id": id_, "type": "EModel"} for id_ in self.emodels]
        emodel_scripts_ids = [{"id": id_, "type": "EModelScript"} for id_ in self.emodel_scripts_id]
        generates = emodels_ids + emodel_scripts_ids
        if self.fitness_configuration_id:
            generates.append(
                {
                    "id": self.fitness_configuration_id,
                    "type": "FitnessCalculatorConfiguration",
                }
            )

        has_part = []
        if self.targets_configuration_id:
            has_part.append(
                {
                    "id": self.targets_configuration_id,
                    "type": "ExtractionTargetsConfiguration",
                }
            )
        if self.pipeline_settings_id:
            has_part.append({"id": self.pipeline_settings_id, "type": "EModelPipelineSettings"})
        if self.emodel_configuration_id:
            has_part.append({"id": self.emodel_configuration_id, "type": "EModelConfiguration"})

        ids = {}
        if generates:
            ids["generates"] = generates
        if has_part:
            ids["hasPart"] = has_part

        return ids

    def as_dict(self):
        """Used for the storage of the object"""
        return vars(self)
