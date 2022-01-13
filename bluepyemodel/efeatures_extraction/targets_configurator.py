"""Targets Configurator"""
import logging

from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.efeatures_extraction.targets_configuration import TargetsConfiguration
from bluepyemodel.emodel_pipeline.utils import yesno

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

    def new_configuration(self, files=None, targets=None, protocols_rheobase=None, auto=False):
        """Create a new configuration"""

        if self.configuration is not None:
            self.delete_configuration()

        if auto:
            self.get_nexus_roaming_base_configuration()
        else:
            self.configuration = TargetsConfiguration(files, targets, protocols_rheobase)

    def load_configuration(self):
        """Load a previously registered configuration"""

        if isinstance(self.access_point, LocalAccessPoint):
            raise Exception("Loading configuration is not yet implemented for local access point")

        self.access_point.get_targets_configuration()

    def save_configuration(self):
        """Save the configuration. The saving medium depends of the access point."""

        if self.configuration:
            self.access_point.store_targets_configuration(self.configuration)

    def delete_configuration(self):
        """Delete the current configuration. Warning: it does not delete the file or resource of
        the configuration."""

        if self.configuration:

            if yesno("Save current configuration ?"):
                self.save_configuration()

            self.configuration = None

    def get_nexus_roaming_base_configuration(self):
        """Overwrite the currently loaded configuration with a new configuration initiated from
        finding available Traces on Nexus"""

        raise Exception("Not implemented yet")
        # self.configuration = TargetsConfiguration()
