"""API using Nexus Forge"""

import logging
import pandas

from kgforge.core import KnowledgeGraphForge  # , Resource

# from kgforge.specializations.resources import Dataset

from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger("__main__")

# pylint: disable=W0231


class Nexus_API(DatabaseAPI):
    """API using Nexus Forge"""

    def __init__(self, forge_path):
        """Init"""
        self.forge = KnowledgeGraphForge(forge_path)
        self.path_dataset = self.forge.paths("Dataset")

    def register(self, resources):
        """Push resources."""
        self.forge.register(resources)

    def fetch(self, type_, conditions):
        """
        Retrieve resources based on conditions.

        Args:
            type_ (str): nexus type of the resources to fetch
            conditions (dict): keys and values used for the WHERE.

        Returns:
            List

        """
        resources = self.forge.search(self.path_dataset.type.id == type_, **conditions)
        return resources

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
        targets_metadata = self.fetch("Extraction_target", {"emodel.name": emodel})

        protocols = {}
        for t in targets_metadata:
            protocols[t.ecode] = {
                "tolerances": t.tolerance,
                "targets": t.targets,
                "efeatures": t.efeatures,
                "location": t.location,
            }

        path_metadata = self.fetch(
            type_="Recording",
            conditions={
                "emodel.name": emodel,
                "emodel.species": species,
                "ecode": tuple(protocols.keys()),
            },
        )

        cells = {}
        for p in path_metadata:

            if p.cell_id not in cells:
                cells[p.cell_id] = {}

            if p.ecode not in cells[p.cell_id]:
                cells[p.cell_id][p.ecode] = []

            trace_metadata = {
                "filepath": p.path,
                "ljp": p.liquid_junction_potential,
            }

            for opt_key in ["ton", "toff", "i_unit", "v_unit", "t_unit"]:
                if (
                    opt_key in vars(p)
                    and getattr(p, opt_key)
                    and not (pandas.isnull(p[opt_key]))
                ):
                    trace_metadata[opt_key] = p[opt_key]

            cells[p["cell_id"]][p["ecode"]].append(trace_metadata)

        return cells, protocols

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
        """ Save a model obtained from BluePyEfe"""

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
