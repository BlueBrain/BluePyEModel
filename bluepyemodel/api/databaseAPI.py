"""Abstract API class."""

import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseAPI:
    """Database API"""

    def __init__(self, emodel):
        """Init"""

        self.emodel = emodel

    def set_emodel(self, emodel):
        """Setter for the name of the emodel."""
        self.emodel = emodel

    def store_efeatures(
        self,
        efeatures,
        current,
        name_Rin_protocol,
        name_rmp_protocol,
        validation_protocols,
    ):
        """Save the efeatures and currents obtained from BluePyEfe"""

    def store_protocols(self, stimuli, validation_protocols):
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
        """Gather the metadata used to build the config dictionary used as an
        input by BluePyEfe.

        Returns:
            cells (dict): return the cells recordings metadata
            protocols (dict): return the protocols metadata

        """

    def get_feature_extraction_config(self, threshold_nvalue_save=1):
        """Get config for feature extraction.

        Args:
            threshold_nvalue_save (int): lower bounds of the number of values required
                to save an efeature.

        Returns:
            extract_config (dict): config dict used in extract.extract_efeatures()
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

    def get_morph_modifiers(self):
        """Get the morph modifiers if any."""

    def get_mechanism_paths(self, mechanism_names):
        """Get the path of the mod files

        Args:
            mechanism_names (list): names of the mechanisms

        Returns:
            mechanism_paths (dict): {'mech_name': 'mech_path'}
        """

    def get_emodel_names(self):
        """Get the list of all the names of emodels

        Returns:
            dict: keys are emodel names with seed, values are names without seed.
        """

    def optimisation_state(self, checkpoint_dir, seed=1, githash=""):
        """Return the state of the optimisation.

        TODO: - should return three states: completed, in progress, empty
              - better management of checkpoints
        """

        checkpoint_path = Path(checkpoint_dir) / f"checkpoint__{self.emodel}__{githash}__{seed}.pkl"

        return checkpoint_path.is_file()

    def _build_pdf_dependencies(self, seed, githash):
        """Find all the pdfs associated to an emodel"""

    def search_figure_path(self, pathname):
        """Search for a single pdf based on an expression"""

        matches = glob.glob(pathname)

        if not matches:
            logger.debug("No pdf for pathname %s", pathname)
            return None

        if len(matches) > 1:
            raise Exception("More than one pdf for pathname %s" % pathname)

        return matches[0]

    def search_figure_efeatures(self, protocol_name, efeature):
        """Search for the pdf representing the efeature extracted from ephys recordings"""

        pdf_amp = self.search_figure_path(f"./{self.emodel}/*{protocol_name}_{efeature}_amp.pdf")

        pdf_amp_rel = self.search_figure_path(
            f"./{self.emodel}/*{protocol_name}_{efeature}_amp_rel.pdf"
        )

        return pdf_amp, pdf_amp_rel

    def search_figure_emodel_optimisation(self, seed, githash=""):
        """Search for the pdf representing the convergence of the optimisation"""

        fname = f"checkpoint__{self.emodel}__{githash}__{seed}.pdf"
        pathname = Path("./figures") / self.emodel / "optimisation" / fname

        return self.search_figure_path(str(pathname))

    def search_figure_emodel_traces(self, seed, githash=""):
        """Search for the pdf representing the traces of an emodel"""

        fname = f"{self.emodel}__{githash}__{seed}_traces.pdf"
        pathname = Path("./figures") / self.emodel / "traces" / "all" / fname

        return self.search_figure_path(str(pathname))

    def search_figure_emodel_score(self, seed, githash=None):
        """Search for the pdf representing the scores of an emodel"""

        if githash:
            fname = f"{self.emodel}_{githash}_{seed}_scores.pdf"
        else:
            fname = f"{self.emodel}_{seed}_scores.pdf"

        pathname = Path("./figures") / self.emodel / "scores" / "all" / fname

        return self.search_figure_path(str(pathname))

    def search_figure_emodel_parameters(self):
        """Search for the pdf representing the distribution of the parameters
        of an emodel"""

        fname = f"{self.emodel}_parameters_distribution.pdf"
        pathname = Path("./figures") / self.emodel / "distributions" / "all" / fname

        return self.search_figure_path(str(pathname))
