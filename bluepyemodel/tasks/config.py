"""Luigi config classes."""

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

import luigi


class EmodelAPIConfig(luigi.Config):
    """Configuration of emodel api database."""

    api = luigi.Parameter(default="local")

    # local parameters
    emodel_dir = luigi.OptionalParameter(default="./")
    recipes_path = luigi.OptionalParameter(default=None)
    final_path = luigi.OptionalParameter(default="./final.json")
    legacy_dir_structure = luigi.BoolParameter(default=False)

    # nexus parameters
    forge_path = luigi.OptionalParameter(default=None)
    forge_ontology_path = luigi.OptionalParameter(default=None)
    nexus_project = luigi.OptionalParameter(default="emodel_pipeline")
    nexus_organisation = luigi.OptionalParameter(default="demo")
    nexus_endpoint = luigi.OptionalParameter(default="https://bbp.epfl.ch/nexus/v1")

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        if self.api == "local":
            if self.recipes_path is None:
                raise ValueError("recipes_path cannot be None when api is set to 'local'")

            self.api_args = {
                "emodel_dir": self.emodel_dir,
                "recipes_path": self.recipes_path,
                "final_path": self.final_path,
                "legacy_dir_structure": self.legacy_dir_structure,
            }

        if self.api == "nexus":
            if self.forge_path is None:
                raise ValueError("forge_path cannot be None when api is set to 'nexus'")
            if self.forge_ontology_path is None:
                raise ValueError("forge_ontology_path cannot be None when api is set to 'nexus'")

            self.api_args = {
                "forge_path": self.forge_path,
                "forge_ontology_path": self.forge_ontology_path,
                "project": self.nexus_project,
                "organisation": self.nexus_organisation,
                "endpoint": self.nexus_endpoint,
            }
