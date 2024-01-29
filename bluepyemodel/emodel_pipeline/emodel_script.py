"""EModelScript class"""

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


class EModelScript:
    """Contains an emodel hoc file path."""

    def __init__(self, hoc_file_path=None, seed=None, workflow_id=None):
        """Init

        Args:
            hoc_file_path (str): path to the hoc file of the emodel
            seed (str): seed used during optimisation for this emodel.
            workflow_id (str): id of the emodel workflow resource
        """
        self.hoc_file_path = hoc_file_path
        self.seed = seed
        self.workflow_id = workflow_id

    def get_related_nexus_ids(self):
        return {
            "generation": {
                "type": "Generation",
                "activity": {
                    "type": "Activity",
                    "followedWorkflow": {"type": "EModelWorkflow", "id": self.workflow_id},
                },
            }
        }

    def as_dict(self):
        return {
            "nexus_distributions": [self.hoc_file_path],
            "seed": self.seed,
        }
