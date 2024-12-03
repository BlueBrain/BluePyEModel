# Author: Yann Roussel <yann.roussel@epfl.ch>
#
# License:
"""
Copyright 2024, EPFL/Blue Brain Project

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

"""
This script upload the output files from the ic_selector_script.py to NEXUS.
Dependencies: pandas, getpass and kgforge
"""

import getpass

from kgforge.core import KnowledgeGraphForge
from kgforge.core import Resource
from kgforge.specializations.resources import Dataset


TOKEN = getpass.getpass()

nexus_endpoint = "https://bbp.epfl.ch/nexus/v1" # production environment

ORG = "bbp"
PROJECT = "ncmv3"

forge = KnowledgeGraphForge("https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml",
                           endpoint=nexus_endpoint,
                           bucket=f"{ORG}/{PROJECT}",
                           token= TOKEN,
                           debug=True
                           )

my_data_distribution = forge.attach("./output/met_type_ion_channel_gene_expression.csv")

# brainLocation as a Resource => so that you can do #my_dataset.brainLocation.brainRegion
brainRegion = Resource(label="Isocortex")
brainLocation = Resource(brainRegion=brainRegion)

my_dataset = Dataset(forge, type=["Entity","Dataset", "RNASequencing"],
                     name="Mouse_met_types_ion_channel_expression",
                     brainLocation = brainLocation,
                     description="Output from IC_selector module"
                    )
my_dataset.add_distribution("./output/met_type_ion_channel_gene_expression.csv")
forge.register(my_dataset)
