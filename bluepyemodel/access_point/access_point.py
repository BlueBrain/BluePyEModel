"""Abstract data access point class."""

import glob
import logging
from pathlib import Path

from bluepyemodel.emodel_pipeline.emodel_settings import EModelPipelineSettings
from bluepyemodel.optimisation import get_checkpoint_path

logger = logging.getLogger(__name__)


class DataAccessPoint:
    """Data access point"""

    def __init__(self, emodel):
        """Init"""

        self.emodel = emodel

    def set_emodel(self, emodel):
        """Setter for the name of the emodel."""
        self.emodel = emodel

    def load_pipeline_settings(self):
        """ """
        return EModelPipelineSettings()

    def store_efeatures(
        self,
        efeatures,
        current,
    ):
        """Save the efeatures and currents obtained from BluePyEfe"""

    def store_protocols(self, stimuli):
        """Save the protocols obtained from BluePyEfe"""

    def store_emodel(
        self,
        scores,
        params,
        optimizer_name,
        seed,
        githash="",
        validated=None,
        scores_validation=None,
    ):
        """Save a model obtained from BluePyOpt"""

    def get_extraction_metadata(self):
        """Get the configuration parameters used for feature extraction.

        Returns:
            files_metadata (dict)
            targets (dict)
            protocols_threshold (list)
        """

    def get_emodel(self):
        """Get dict with parameter of single emodel (including seed if any)"""

    def get_emodels(self, emodels):
        """Get the list of emodels dictionaries."""

    def get_parameters(self):
        """Get the definition of the parameters to optimize as well as the
         locations of the mechanisms. Also returns the name to the mechanisms.

        Returns:
            params_definition (dict):
            mech_definition (dict):
            mech_names (list):

        """

    def get_protocols(self, include_validation=False):
        """Get the protocols from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            include_validation (bool):should the validation protocols be added to the evaluator.

        Returns:
            protocols_out (dict): protocols definitions
        """

    def get_features(self, include_validation=False):
        """Get the efeatures from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            include_validation (bool): should the validation efeatures be added to the evaluator.

        Returns:
            efeatures_out (dict): efeatures definitions
        """

    def get_morphologies(self):
        """Get the name and path to the morphologies.

        Returns:
            morphology_definition (dict): [{'name': morph_name,
                                            'path': 'morph_path'}

        """

    def download_mechanisms(self):
        """Download the mod files when not already downloaded"""

    def get_emodel_names(self):
        """Get the list of all the names of emodels

        Returns:
            dict: keys are emodel names with seed, values are names without seed.
        """

    def optimisation_state(self, seed=1, githash=""):
        """Return the state of the optimisation.

        TODO: - should return three states: completed, in progress, empty
              - better management of checkpoints
        """

        checkpoint_path = get_checkpoint_path(self.emodel, seed, githash)

        return checkpoint_path.is_file()

    def _build_pdf_dependencies(self, seed, githash):
        """Find all the pdfs associated to an emodel"""
