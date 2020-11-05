"""Generic API class."""


class DatabaseAPI:
    """Database API"""

    def __init__(self):
        """"""

    def store_efeatures(self, emodel, species, efeatures, current):
        """ Save the efeatures and currents obtained from BluePyEfe"""

    def store_protocols(self, emodel, species, stimuli):
        """ Save the protocols obtained from BluePyEfe"""

    def store_model(
        self,
        emodel,
        species,
        scores,
        params,
        optimizer_name,
        seed=None,
        validated=False,
    ):
        """ Save a model obtained from BluePyOpt"""

    def get_extraction_metadata(self, emodel, species):
        """Gather the metadata used to build the config dictionary used as an
        input by BluePyEfe.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            cells (dict): return the cells recordings metadata
            protocols (dict): return the protocols metadata

        """

    def get_parameters(self, emodel, species):
        """Get the definition of the parameters to optimize as well as the
         locations of the mechanisms. Also returns the name to the mechanisms.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            params_definition (dict):
            mech_definition (dict):
            mech_names (list):

        """

    def get_protocols(
        self,
        emodel,
        species,
        delay=0.0,
        include_validation=False
    ):
        """Get the protocols from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            delay (float): additional delay in ms to add at the start of
                the protocols.

        Returns:
            protocols_out (dict): protocols definitions

        """

    def get_features(
        self,
        emodel,
        species,
        include_validation=False
    ):
        """Get the efeatures from the database and put in a format that fits
         the MainProtocol needs.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            efeatures_out (dict): efeatures definitions

        """

    def get_morphologies(self, emodel, species):
        """Get the name and path to the morphologies.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            morphology_definition (dict): [{'name': morph_name,
                                            'path': 'morph_path'}

        """

    def get_mechanism_paths(self, mechanism_names):
        """Get the path of the mod files

        Args:
            mechanism_names (list): names of the mechanisms

        Returns:
            mechanism_paths (dict): {'mech_name': 'mech_path'}

        """

    def close(self):
        """Close the database."""
