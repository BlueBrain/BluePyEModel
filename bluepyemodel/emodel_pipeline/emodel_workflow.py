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

        ids = {
            "generates": generates,
            "hasPart": [
                {"id": self.targets_configuration_id, "type": "TargetsConfiguration"},
                {"id": self.pipeline_settings_id, "type": "EModelPipelineSettings"},
                {"id": self.emodel_configuration_id, "type": "NeuronModelConfiguration"},
            ],
        }

        return ids

    def as_dict(self):
        """Used for the storage of the object"""
        return vars(self)
