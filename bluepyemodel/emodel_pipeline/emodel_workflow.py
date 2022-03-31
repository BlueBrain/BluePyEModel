"""EModelWorkflow class"""
import logging

logger = logging.getLogger(__name__)


class EModelWorkflow:
    """Contains the state of the workflow and the configurations needed for the workflow"""

    def __init__(
        self,
        targets_configuration_id,
        pipeline_settings_id,
        emodel_configuration_id,
        state="not launched",
    ):
        """Init

        Args:
            targets_configuration (str): TargetsConfiguration nexus id
            pipeline_settings (str): EModelPipelineSettings nexus id
            emodel_configuration (str): NeuronModelConfiguration id
            state (str): can be "not launched", "running" or "done"
        """
        self.targets_configuration_id = targets_configuration_id
        self.pipeline_settings_id = pipeline_settings_id
        self.emodel_configuration_id = emodel_configuration_id
        self.state = state

    def get_configuration_ids(self):
        """Return all configuration id parameters"""
        return (
            self.targets_configuration_id,
            self.pipeline_settings_id,
            self.emodel_configuration_id,
        )

    def as_dict(self):
        """Used for the storage of the object"""
        return vars(self)
