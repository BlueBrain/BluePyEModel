"""Methods for selecting genes associated with a given me- and t-type"""

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


import logging

import numpy as np

from .distribution import Distribution


class GeneSelector:
    """Selects genes from Yann's gene mapping file."""

    def __init__(self, gene_map):
        """
        Args:
            gene_map (DataFrame): Array mapping genes to met-types. Rows have
            met-types, columns have genes
        """
        self._gene_map = gene_map
        self.selected_genes = {}
        self.selected_met_types = []

    @staticmethod
    def _filter(df, keys):
        """Rules for filtering the columns of Yann's gene map.

        Args:
            df (DataFrame): subset of Yann's table
            keys (list [str]): list of strings to filter columns

        Returns:
            df (DataFrame): subset of the table
        """

        # First filter on exact match
        for name in df.index.names:
            crit = df.index.get_level_values(name).isin(keys)
            # Only apply filter if exact match was found
            if not np.all(~crit):
                df = df[crit]
        # Then filter on partial match
        for key in keys:
            new_df = df.filter(regex=key, axis=0)
            # Only apply filter if partial match was found
            if not new_df.shape[0] == 0:
                df = new_df
        return df

    @staticmethod
    def _get_gene_presence(gene):
        """Determine if a gene is present or not.

        Args:
            gene (DataFrame): single column from the gene table

        Returns:
            presence (array [bool]): gene is present
        """

        gene_present = gene["presence"]
        if isinstance(gene_present, str):
            presence = np.array([int(gene_present)], dtype=int) == 1
        else:
            presence = np.array(gene_present.values, dtype=int) == 1
        return presence

    @staticmethod
    def _get_distributions(gene, presence, group_compartments):
        """Get distribution per compartment for a given gene.

        Args:
            gene (DataFrame): single column from the gene table
            presence (array [bool]): gene is present
            group_compartments (bool): use 'all' if present in all compartments

        Returns:
            distros (Distribution): gene distribution per compartment
        """

        all_comps = None
        dend = np.unique(np.array(gene.filter(regex="dend", axis=0).values[presence], dtype=str))
        axon = np.unique(np.array(gene.filter(regex="axon", axis=0).values[presence], dtype=str))
        soma = np.unique(np.array(gene.filter(regex="soma", axis=0).values[presence], dtype=str))

        if len(dend) > 1:
            logging.warning(
                "Conflicting dendritic distributions for gene %s. Selected '%s' from %s",
                gene.name,
                dend[0],
                dend,
            )
        dend = dend[0]
        if len(axon) > 1:
            logging.warning(
                "Conflicting axonal distributions for gene %s. Selected '%s' from %s",
                gene.name,
                axon[0],
                axon,
            )
        axon = axon[0]
        if len(soma) > 1:
            logging.warning(
                "Conflicting somatic distributions for gene %s. Selected '%s' from %s",
                gene.name,
                soma[0],
                soma,
            )
        soma = soma[0]

        if (dend == axon) and (axon == soma):
            all_comps = dend

        def str_has_value(s):
            return not ((s == "nan") or (s.strip() == ""))

        distros = Distribution()
        if all_comps and group_compartments:
            if str_has_value(all_comps):
                distros.all = all_comps
        else:
            distros.set_fields(somatic=soma, axonal=axon, basal=dend, apical=dend)
        return distros

    @staticmethod
    def _get_gbar_max(gene):
        """Get gbar_max for given gene.

        Args:
            gene (DataFrame): single column from the gene table

        Returns:
            gbar_max (float): gbar_max value
        """

        try:
            gbar_max = gene["g_bar_max"]
            if isinstance(gbar_max, (str, float)):
                gbar_max = np.array([float(gbar_max)])
            else:
                gbar_max = np.array([float(g) for g in gbar_max.values])
        except KeyError:
            gbar_max = np.array([0.0])
        crit = np.array([np.isnan(g) for g in gbar_max])
        gbar_max[crit] = 0.0
        gbar_max = np.unique(gbar_max)
        if len(gbar_max) > 1:
            logging.warning(
                "Conflicting g_bar_max values for gene %s. Selected '%s' from %s",
                gene.name,
                np.max(gbar_max),
                gbar_max,
            )
        return np.max(gbar_max)

    def get(self, gene_name):
        """Get info for given gene.

        Args:
            gene_name (str): name of gene

        Returns:
            (dict): gene info
        """

        if gene_name in self.selected_genes:
            return self.selected_genes[gene_name]
        raise KeyError("Gene not available for the selected ttype.")

    def select_from_ttype(self, keys=None, group_compartments=False):
        """Returns selected genes and distributions associated with provided
        key words.

        Args:
            keys (list [str]): List of keywords
            group_compartments: Option to combine compartments into groups
            (e.g. 'all', or 'alldend')

        Returns:
            selected_genes (list [dict]): list of selected genes with info
        """

        df = self._gene_map
        if isinstance(keys, str):
            keys = [keys]
        if keys:
            df = self._filter(df, keys)
        # Store result
        self.selected_met_types = np.unique([f"{v[0]} - {v[1]}" for v in df.index.values])
        df = df.droplevel([0, 1])
        # Apply filter also to genes
        if keys:
            genes = self._filter(df.T, keys)
            genes = genes.T
        else:
            genes = df

        names = genes.columns.values
        for name in names:
            gene = genes[name]
            # Check if gene is present
            presence = self._get_gene_presence(gene)
            if sum(presence) == 0:
                continue
            # Determine distributions per compartment
            distros = self._get_distributions(gene, presence, group_compartments)
            # Determine gbar_max
            gbar_max = self._get_gbar_max(gene)
            # Add item
            self.selected_genes[name] = {
                "channel": "n/a",
                "distribution": distros,
                "gbar_max": gbar_max,
            }

        return self.selected_genes

    def __str__(self):
        heading_str = "\n>>> M/E/T types <<<"
        out_str = [heading_str]
        for k in self.selected_met_types:
            out_str += [k]
        out_str += ["\n>>> Genes <<<"]
        items = self.selected_genes.items()
        n = 1
        for k, v in items:
            items_str = f"-{n}- {k}, distribution: {v['distribution']}, gbar_max: {v['gbar_max']}"
            n += 1
            out_str += [items_str]
        return "\n".join(out_str)
