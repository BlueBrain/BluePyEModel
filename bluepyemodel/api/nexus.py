"""API using Nexus Forge"""

import logging

from kgforge.core import KnowledgeGraphForge  # , Resource
from bluepyemodel.api.databaseAPI import DatabaseAPI

logger = logging.getLogger("__main__")

# pylint: disable=W0231,W0221


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
            conditions (dict): keys and values used for the "WHERE".

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
        protocols = {}
        cells = {}
        protocols_threshold = []

        # Fetch the data from the targets for E-features extraction

        return cells, protocols, protocols_threshold

    def store_efeatures(self, emodel, species, efeatures, currents):
        """Store the efeatures and currents obtained from BluePyEfe in
            the extracted e-features ressources.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            efeatures (dict): of the format:
                                {
                                    'protocol_name':[
                                        {'feature': feature_name, value: [mean, std]},
                                        {'feature': feature_name2, value: [mean, std]}
                                    ]
                                }
            currents (dict): of the format:
                                {
                                    "hypamp": [mean, std],
                                    "thresh": [mean, std]
                                }

        """

    def store_protocols(self, emodel, species, stimuli):
        """Store the protocols obtained from BluePyEfe in
            the extracted protocols ressources.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            stimuli (dict): of the format:
                                {
                                    'protocol_name':
                                        {"step": ..., "holding": ...}
                                }
        """

    def store_emodel(
        self,
        emodel,
        scores,
        params,
        optimizer_name,
        seed,
        validated=False,
        species=None,
    ):
        """Store an emodel obtained from BluePyOpt in a EModel ressource

        Args:
            emodel (str): name of the emodel
            scores (dict): scores of the efeatures,
            params (dict): values of the parameters,
            optimizer_name (str): name of the optimizer.
            seed (int): seed used for optimization,
            validated (bool): True if the model has been validated.
            species (str): name of the species (rat, human, mouse)
        """

    def get_emodels(self, emodels, species):
        """Get the list of emodels matching the .

        Args:
            emodels (list): list of names of the emodel's metypes
            species (str): name of the species (rat, human, mouse)


        Returns:
            models (list): return the emodels, of the format:
            [
                {
                    "emodel": ,
                    "species": ,
                    "fitness": ,
                    "parameters": ,
                    "scores": ,
                    "validated": ,
                    "optimizer": ,
                    "seed": ,
                }
            ]
        """
        models = []
        return models

    def get_parameters(self, emodel, species):
        """Get the definition of the parameters to optimize from the
            optimization parameters ressources, as well as the
            locations of the mechanisms. Also returns the names of the mechanisms.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            params_definition (dict): of the format:
                definitions = {
                        'distributions':
                            {'distrib_name': {
                                'function': function,
                                'parameters': ['param_name']}
                             },
                        'parameters':
                            {'sectionlist_name': [
                                    {'name': param_name1, 'val': [lbound1, ubound1]},
                                    {'name': param_name2, 'val': 3.234}
                                ]
                             }
                    }
            mech_definition (dict): of the format:
                mechanisms_definition = {
                    section_name1: {
                        "mech":[
                            mech_name1,
                            mech_name2
                        ]
                    },
                    section_name2: {
                        "mech": [
                            mech_name3,
                            mech_name4
                        ]
                    }
                }
            mech_names (list): list of mechanisms names

        """

        params_definition = {"distributions": {}, "parameters": {}}
        mech_definition = {}
        mech_names = []

        return params_definition, mech_definition, mech_names

    def get_protocols(self, emodel, species, delay=0.0, include_validation=False):
        """Get some of the protocols from the "Extracted protocols" resources depending
            on "Optimization and validation targets" ressources.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            delay (float): additional delay in ms to add at the start of
                the protocols.
            include_validation (bool): if True, returns the protocols for validation as well

        Returns:
            protocols_out (dict): protocols definitions. Of the format:
                {
                     protocolname: {
                         "type": "StepProtocol" or "StepThresholdProtocol",
                         "stimuli": {...}
                         "extra_recordings": ...
                     }
                }
        """
        protocols_out = {}

        return protocols_out

    def get_features(
        self,
        emodel,
        species,
        include_validation=False,
    ):
        """Get the efeatures from the "Extracted e-features" ressources  depending
            on "Optimization and validation targets" ressources.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)
            include_validation (bool): should the features for validation be returned as well

        Returns:
            efeatures_out (dict): efeatures definitions. Of the format:
                {
                    "protocol_name": {"soma.v":
                        [{"feature": feature_name, val:[mean, std]}]
                    }
                }

        """
        efeatures_out = {
            "RMPProtocol": {"soma.v": []},
            "RinProtocol": {"soma.v": []},
            "SearchHoldingCurrent": {"soma.v": []},
            "SearchThresholdCurrent": {"soma.v": []},
        }

        return efeatures_out

    def get_morphologies(self, emodel, species):
        """Get the name and path (or data) to the morphologies from
            the "Optimization morphologies" ressources.

        Args:
            emodel (str): name of the emodel
            species (str): name of the species (rat, human, mouse)

        Returns:
            morphology_definition (list): [{'name': morph_name,
                                            'path': 'morph_path'}

        """
        morphology_definition = []

        return morphology_definition

    def get_mechanism_paths(self, mechanism_names):
        """Get the path of the mod files from the "Mechanisms" ressrouce,
            to copy and compile them locally.

        Args:
            mechanism_names (list): names of the mechanisms

        Returns:
            mechanism_paths (dict): {'mech_name': 'mech_path'}

        """
        mechanism_paths = {}

        return mechanism_paths
