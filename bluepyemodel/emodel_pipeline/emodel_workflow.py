"""EModelWorkflow class"""

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

logger = logging.getLogger(__name__)


class EModelWorkflow:
    """Contains the state of the workflow and the configurations needed for the workflow"""

    def __init__(
        self,
        targets_configuration_id,
        pipeline_settings_id,
        emodel_configuration_id,
        emodels=None,
        state="not launched",
    ):
        """Init

        Args:
            targets_configuration (str): TargetsConfiguration nexus id
            pipeline_settings (str): EModelPipelineSettings nexus id
            emodel_configuration (str): NeuronModelConfiguration id
            emodels (list): list of EModel ids
            state (str): can be "not launched", "running" or "done"
        """
        self.targets_configuration_id = targets_configuration_id
        self.pipeline_settings_id = pipeline_settings_id
        self.emodel_configuration_id = emodel_configuration_id
        self.emodels = emodels if emodels else []
        self.state = state

    def add_emodel_id(self, emodel_id):
        """Add an emodel id to the list of emodels"""
        self.emodels.append(emodel_id)

    def get_configuration_ids(self):
        """Return all configuration id parameters"""
        ids = (
            self.targets_configuration_id,
            self.pipeline_settings_id,
            self.emodel_configuration_id,
        )
        if self.emodels:
            ids += tuple(self.emodels)
        return ids

    def get_related_nexus_ids(self):
        generates = [{"id": id_, "type": "EModel"} for id_ in self.emodels]

        has_part = []
        if self.targets_configuration_id:
            has_part.append(
                {"id": self.targets_configuration_id, "type": "ExtractionTargetsConfiguration"}
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
