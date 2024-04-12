"""Targets Configurator"""

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
from bluepyemodel.efeatures_extraction.auto_targets import get_auto_target_from_presets
from bluepyemodel.efeatures_extraction.targets_configuration import TargetsConfiguration
from bluepyemodel.tools.utils import yesno

logger = logging.getLogger(__name__)


class TargetsConfigurator:
    """Handles the loading, saving and modification of a targets configuration"""

    def __init__(self, access_point, configuration=None):
        """Creates a targets configuration, which includes the ephys files and targets

        Args:
            access_point (DataAccessPoint): access point to the emodel data.
            configuration (TargetsConfiguration): a pre-existing configuration.
        """

        self.access_point = access_point
        self.configuration = configuration

    def new_configuration(
        self,
        files=None,
        targets=None,
        protocols_rheobase=None,
        auto_targets=None,
        protocols_mapping=None,
    ):
        """Create a new configuration"""

        if self.configuration is not None:
            self.delete_configuration()

        available_traces = self.access_point.get_available_traces()
        available_efeatures = self.access_point.get_available_efeatures()

        self.configuration = TargetsConfiguration(
            files=files,
            targets=targets,
            protocols_rheobase=protocols_rheobase,
            available_traces=available_traces,
            available_efeatures=available_efeatures,
            auto_targets=auto_targets,
            protocols_mapping=protocols_mapping,
        )

    def load_configuration(self):
        """Load a previously registered configuration"""

        if isinstance(self.access_point, LocalAccessPoint):
            raise NotImplementedError(
                "Loading configuration is not yet implemented for local access point"
            )

        self.access_point.get_targets_configuration()

    def save_configuration(self):
        """Save the configuration. The saving medium depends of the access point."""

        if self.configuration:
            if self.configuration.is_configuration_valid:
                self.access_point.store_targets_configuration(self.configuration)
            else:
                raise ValueError("Couldn't save invalid configuration")

    def delete_configuration(self):
        """Delete the current configuration. Warning: it does not delete the file or resource of
        the configuration."""

        if self.configuration:
            if yesno("Save current configuration ?"):
                self.save_configuration()

            self.configuration = None

    def create_and_save_configuration_from_access_point(self):
        """Create and save a new configuration given data from access point."""
        # TODO: this function does not handle additional fitness protocols and efeatures yet.
        if self.access_point.has_targets_configuration():
            logger.info(
                "Targets configuration already present on access point."
                "Will not create another one."
            )
            return

        files = self.access_point.pipeline_settings.files_for_extraction
        targets = self.access_point.pipeline_settings.targets
        protocols_rheobase = self.access_point.pipeline_settings.protocols_rheobase
        protocols_mapping = self.access_point.pipeline_settings.protocols_mapping

        if not targets:
            auto_targets = self.access_point.pipeline_settings.auto_targets
            auto_targets_presets = self.access_point.pipeline_settings.auto_targets_presets
            if not auto_targets:
                if not auto_targets_presets:
                    raise TypeError(
                        "Please fill either targets, auto_targets or auto_targets_presets."
                        "Or alternatively save your targets before running your pipeline."
                    )
                auto_targets = get_auto_target_from_presets(auto_targets_presets)

        self.new_configuration(files, targets, protocols_rheobase, auto_targets, protocols_mapping)
        self.save_configuration()
