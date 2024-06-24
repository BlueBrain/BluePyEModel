""" ICSELECTOR
Select a set of NEURON mechanisms, parameters and bounds from corresponding
genes. Gene names can be selected from a file mapping genes to different
ME-T types. Corresponding channels are selected from a file mapping channels
and parameters to genes.
"""

"""
Copyright 2023-2024, EPFL/Blue Brain Project

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


import json
import logging

import numpy as np
import pandas as pd

from .modules.configuration import Configuration
from .modules.gene_selector import GeneSelector
from .modules.model_selector import ModelSelector


class ICSelector:
    """Selects NEURON mechanisms, with parameters, value bounds, and
    distributions. Uses reference files for MET-type to gene mapping,
    gene to channel mapping, and channel to parameter mapping."""

    def __init__(self, ic_map_path, gene_map_path, mode="mixed", status="latest"):
        """
        Args:
            ic_map_path (str): path to ic_mapping file (.json)
            gene_map_path (str): path to gene mapping file (.csv)
            mode (str): types of ion channel model to select.
                Options are 'generic', 'genetic', and 'mixed' (default).
            status (str): model status. Options are 'stable' and 'latest'.
        """

        # === Load mapping files
        gene_map = pd.read_csv(gene_map_path, index_col=[0, 1, 2])
        self._gene_selector = GeneSelector(gene_map)

        with open(ic_map_path, mode="r", encoding="utf-8") as fid:
            ic_map = json.load(fid)
        self._misc_parameters = ic_map["misc_parameters"]
        self._gene_to_ic = ic_map["genes"]
        self._distributions = ic_map["distributions"]
        self._model_selector = ModelSelector(ic_map)

        self.set_status(status)
        self.set_mode(mode)

    def set_status(self, status):
        """Select model types.

        Args:
            status (str): 'stable' or 'latest'
        """
        if status in ["stable", "latest"]:
            self._model_selector.status = status

    def set_mode(self, mode):
        """Select configuration mode.

        Args:
            mode (str): 'generic', 'genetic', or 'mixed'
        """

        if mode in ["genetic", "generic", "mixed"]:
            self._model_selector.mode = mode

    def _get_channel_name(self, gene_name):
        """Get channel name from gene.

        Args:
            gene_name (str): gene name

        Returns:
            (str): channel name
        """
        if gene_name in self._gene_to_ic:
            return self._gene_to_ic[gene_name]
        return None

    def _get_mech_from_gene(self, gene_name):
        """Get channel mechanism from gene.

        Args:
            gene_name (str): gene name

        Returns:
            (Mechanism): channel mechanism
        """

        channel_name = self._get_channel_name(gene_name)
        if channel_name:
            return self._model_selector.get(channel_name)
        return None

    def __set_parameters_from_gene(self, gene_name):
        """Set mechanism fields corresponding to gene.
        Args:
            gene_name (str): gene name
        """

        info = self._gene_selector.get(gene_name)
        channel_name = info["channel"]
        mech = self.get(channel_name)
        if mech:
            mech.set_from_gene_info(info)
        else:
            logging.warning("Could not determine mechanism for gene %s", gene_name)

    def _set_parameters_from_ttype(self):
        """Copy parameters from selected genes to mechanims."""

        for gene_name in self._gene_selector.selected_genes:
            self.__set_parameters_from_gene(gene_name)

    def _select_genes_from_ttype(self, key_words):
        """Select genes from key words.
        Args:
            key_words (list [str]): list of keys to select genes
        """

        # === Get genes from gene mapping file
        logging.info("\n===============\nGenes Selection\n===============")
        genes = self._gene_selector.select_from_ttype(key_words)
        logging.info(str(self._gene_selector))

        # === Map genes to channels
        for gene, info in genes.items():
            if gene in self._gene_to_ic:
                info["channel"] = self._gene_to_ic[gene]

    def _select_mechanisms_from_ttype(self):
        """Select mechanisms from previously selected genes."""

        logging.info("\n==================\nChannels Selection\n==================")
        for gene_name in self._gene_selector.selected_genes:
            self.select(gene_name)
        logging.info(str(self._model_selector))

    def get(self, name):
        """Get mechanism corresponding to gene or channel.

        Args:
            name (str): name of a gene or channel.

        Returns:
            mech (Mechanism): selected mechanism
        """

        # Try first to select as channel name
        mech = self._model_selector.get(name)
        if not mech:
            # If that fails, try as gene name
            mech = self._get_mech_from_gene(name)
        return mech

    def select(self, name):
        """Select mechanism for inclusion in configuration.

        Args:
            name (str): name of a gene or channel.
        """

        selected = self._model_selector.select(name)
        if not selected:
            channel_name = self._get_channel_name(name)
            selected = self._model_selector.select(channel_name)

    def get_gene_mapping(self):
        """Get a dict of all selected genes and mapped channels.
        Returns:
            genes (dict): all selected genes and mapped channels
        """

        mechs = self.get_mechanisms()
        genes = self._gene_selector.selected_genes
        genes = {k: v for k, v in genes.items() if not v["channel"] == "n/a"}
        for k, v in genes.items():
            channel = v["channel"]
            for a, b in mechs.items():
                if channel in b["_mapped_from"]:
                    v["mapped_to"] = a
        return genes

    def get_mechanisms(self, selected_only=True):
        """Get all available mechanisms from the icmapping file.

        Args:
            selected_only (bool): flag to get only selected channels

        Returns:
            mechs (dict [Mechanism]): mechanisms with all associated info
                fields
        """

        mechs = self._model_selector.get_mechanisms(selected_only)
        mechs = {k: v.asdict() for k, v in mechs.items()}
        return mechs

    def get_selected_cell_types(self):
        """Get met types selected from the gene mapping table.

        Returns:
            (list): all selected cell types
        """

        return self._gene_selector.selected_met_types

    def _get_cell_config(self):
        """Generate cell model configuration from selected mechanisms.

        Returns:
            config (Configuration): set of mechanism parameters per compartment
        """

        mechanisms = self._model_selector.get_mechanisms()
        config = Configuration()
        for mech in mechanisms.values():
            config.add_from_mechanism(mech)

        logging.info("\n=======================\nParameter Configuration\n=======================")

        # === Set additional parameters
        misc = self._misc_parameters
        for name, setting in misc.items():

            if "local" in setting:
                # Location to be obtained from mechanism
                locations = []
                for mech in mechanisms.values():
                    model = mech.model
                    if "read" in model:
                        reads = model["read"]
                        if name in reads:
                            locs = mech.distribution.get()
                            locations.extend(locs.keys())
                for loc in np.unique(locations):
                    config.add_parameter(name, loc, setting["local"])
            else:
                for loc, value in setting.items():
                    config.add_parameter(name, loc, value)

        logging.info(str(config))
        return config

    def get_cell_config_from_ttype(self, key_words):
        """Get all information related to cell model configuration from
        mechanisms selected based on genetic expression profiles.

        Args:
            key_words: keys to select MET-types

        Returns:
            parameters (list [dict,]): mechanism parameters per compartment
            mechanisms (list [dict,]): mechanisms per compartment
            distributions (list [dict,]): definitions of subcellular distributions
            nexus_keys (list [dict,]): name and modelid for each mechanism
        """

        # Perform selections
        self._select_genes_from_ttype(key_words)
        self._set_parameters_from_ttype()
        self._select_mechanisms_from_ttype()

        # Get configuration
        mechs = self._model_selector.get_mechanisms()
        config = self._get_cell_config()

        # Get configuration parameters
        parameters = config.get_parameters()
        mechanisms = config.get_mechanisms()
        distributions = config.get_distributions()

        # Define distributions
        distr_funcs = self._distributions
        for distr in distributions:
            distr_name = distr["name"]
            if not distr_name == "uniform":
                if distr_name in distr_funcs:
                    d = distr_funcs[distr_name]
                    for k, v in d.items():
                        distr[k] = v
                else:
                    logging.warning("Unknown distribution '%s'.", distr_name)

        # Get Nexus keys
        nexus_keys = [v.nexus for k, v in mechs.items() if v.nexus]

        return parameters, mechanisms, distributions, nexus_keys
