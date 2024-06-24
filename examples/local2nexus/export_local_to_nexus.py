"""Upload local models to nexus."""

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

import getpass
import os
from bluepyemodel.export_emodel.export_emodel import export_emodels_nexus
from bluepyemodel.access_point.local import LocalAccessPoint
import logging

logger = logging.getLogger(__name__)

# Please change the following settings according to your needs and data:
emodel = "YOUR_EMODEL_NAME_HERE"  # name of the local emodel you want to upload as it appears in the config/recipes.json file
githash = (
    "YOUR_GITHASH_HERE"  # githash of the local emodel (provided during the optimization step).
)
only_validated = False  # only upload validated emodels

only_best = False  # only upload best emodel
seeds = [1]  # list of seeds that you want to upload, Please leave it empty if only_best=True.

# should match the data of your LocalAccessPoint emodel, if it was not set, use None
etype = None
mtype = None
ttype = None
species = "rat"  # e.g. "mouse"
brain_region = "SSCX"  # e.g. "SSCX"

description = ""

# Nexus settings
nexus_project = ""  # a valid Nexus project name to which the emodel should be uploaded.
nexus_organisation = "bbp"  # choose between "bbp" or "public"
# Nexus advanced settings (only change if you know what you are doing)
nexus_endpoint = "https://bbp.epfl.ch/nexus/v1"
forge_path = "./forge.yml"
forge_ontology_path = "./nsg.yml"
sleep_time = 10  # increase the delay in case indexing is slow


def main():

    # Set the logging level
    logging.basicConfig(level=logging.INFO)

    print("Please input your access token you got from nexus:")
    access_token = getpass.getpass()

    local_access_point = LocalAccessPoint(
        emodel=emodel,
        etype=etype,
        mtype=mtype,
        ttype=ttype,
        species=species,
        brain_region=brain_region,
        final_path=os.path.join(".", "final.json"),
        iteration_tag=githash,
        recipes_path=os.path.join(".", "config", "recipes.json"),
    )

    export_emodels_nexus(
        local_access_point,
        nexus_organisation=nexus_organisation,
        nexus_project=nexus_project,
        nexus_endpoint=nexus_endpoint,
        forge_path=forge_path,
        access_token=access_token,
        only_validated=only_validated,
        only_best=only_best,
        seeds=seeds,
        description=description,
        forge_ontology_path=forge_ontology_path,
        sleep_time=sleep_time,
    )


if __name__ == "__main__":
    main()
