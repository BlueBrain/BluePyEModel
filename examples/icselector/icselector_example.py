""" ICSELECTOR
Select a set of NEURON mechanisms, parameters and bounds from corresponding
genes. Gene names can be selected from a file mapping genes to different
ME-T types. Corresponding channels are selected from a file mapping channels
and parameters to genes.

## Usage
Select from a gene-mapping file:
    $ python icselector.py
        --ic_map <ic_mapping_file.json>
        --gene_map <gene_mapping_file.csv>
        --keys <any_key_from_gene_map> ...

<any_key_from_gene_map> could be an me-type, t-type or gene name from the
gene_map file, or part of a name e.g.
        --keys L3_TPC:A 'L2/3 IT Cxcl14_1'
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


from argparse import ArgumentParser
from pprint import pprint

from bluepyemodel.icselector import ICSelector


# Get command line arguments
parser = ArgumentParser(
    description="Retrieve a list of NEURON mechanisms associated with provided genes.",
    usage="%(prog)s",
)
parser.add_argument(
    "--ic_map",
    dest="ic_map_path",
    type=str,
    help="Path to .json file containing gene to channel mapping.",
)
parser.add_argument(
    "--gene_map",
    dest="gene_map_path",
    type=str,
    help="Path to .csv file containing met-type to gene mapping.",
)
parser.add_argument(
    "--keys",
    type=str,
    nargs="+",
    help="Optional list of keywords to filter genes.",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["genetic", "generic", "mixed"],
    default="mixed",
    help="Model types to include.",
)
parser.add_argument(
    "--status",
    type=str,
    choices=["stable", "latest"],
    default="latest",
    help="Model version to include.",
)
args = parser.parse_args()
cmd_args = vars(args)

# === Instantiate ICSelector
keys = cmd_args.pop("keys", None)
icselector = ICSelector(**cmd_args)

# === Get cell configuration
parameters, mechanisms, distributions, nexus_keys = icselector.get_cell_config_from_ttype(keys)
pprint(parameters)
pprint(mechanisms)
pprint(distributions)
pprint(nexus_keys)

# === Retrieve all mechanisms available in Nexus
mechs = icselector.get_mechanisms(selected_only=True)
pprint(mechs)

# === Retrieve channels mapped from genes
genes = icselector.get_gene_mapping()
pprint(genes)
