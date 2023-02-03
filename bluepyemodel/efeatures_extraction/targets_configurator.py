"""Targets Configurator"""
import logging

from bluepyemodel.access_point.local import LocalAccessPoint
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

    def new_configuration(self, files=None, targets=None, protocols_rheobase=None):
        """Create a new configuration"""

        if self.configuration is not None:
            self.delete_configuration()

        available_traces = self.access_point.get_available_traces()
        available_efeatures = self.access_point.get_available_efeatures()

        self.configuration = TargetsConfiguration(
            files, targets, protocols_rheobase, available_traces, available_efeatures
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
