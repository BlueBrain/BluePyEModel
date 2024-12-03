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
This script download from NEXUS all the required input files for the ic_selector_script.py.
Dependencies: pandas, getpass and kgforge
"""

import getpass

from kgforge.core import KnowledgeGraphForge

print("enter your Nexus password")
TOKEN = getpass.getpass()
# TOKEN = ""
print("password taken")

nexus_endpoint = "https://bbp.epfl.ch/nexus/v1" # production environment

ORG = "bbp"
PROJECT = "ncmv3"

forge = KnowledgeGraphForge("https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml",
                           endpoint=nexus_endpoint,
                           bucket=f"{ORG}/{PROJECT}",
                           token= TOKEN,
                           debug=True
                           )
name_list = ["mouse-whole-cortex-and-hippocampus-smart-seq", "BBP_mtype_list",
             "P(marker_BBPmetype)_L1", "P(marker_BBPmetype)_L23_L6"]

for name in name_list:
    print(name)
    filters = {"type":"Dataset", "name":name}
    results = forge.search(filters, limit=3)
    print(f"{len(results)} results found")

    forge.download(results, "distribution.contentUrl", path="./input/")
    print("________")