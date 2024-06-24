"""Running the emodel with BlueCelluLab"""

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

import argparse
import getpass
import glob
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from kgforge.core import KnowledgeGraphForge

from bluepyemodel.access_point.nexus import NexusAccessPoint

######################################################
# CONFIGURATION
######################################################
# Amplitudes for simulation
# if using threshold based amplitudes, set the amplitudes as a percentage of the threshold current
amplitudes = [-120, -40, 0, 150, 200, 250] # threshold based
# amplitudes = [-0.2, -0.4, -0.6, -0.8, -1.0, 0.2, 0.4, 0.6, 0.8, 1.0] # absolute amplitudes

TEMPERATURE = 34.0 # celsius
V_INIT = -70 # mV

# Nexus configuration
ORG = "" # "bbp" or "public
PROJECT = "" # Nexus project name where the emodel is stored
bucket = f"{ORG}/{PROJECT}"
endpoint = "https://bbp.epfl.ch/nexus/v1"
access_token = getpass.getpass("Enter your Nexus token: ")
forge_path = (
    "https://raw.githubusercontent.com/BlueBrain/nexus-forge/"
    + "master/examples/notebooks/use-cases/prod-forge-nexus.yml"
)
######################################################

def getHoldingThreshCurrent(directory_path):
    pattern = os.path.join(directory_path, 'EM_*' + 'json')
    final = glob.glob(pattern)
    if final:
        file_name = final[0]
        with open(file_name, 'r') as file:
            data = json.load(file)
    else:
        raise FileNotFoundError(f"No EModel resource found in {directory_path}.")

    holding_current = 0
    threshold_current = 0

    for feature in data['features']:
        if 'soma.v.bpo_holding_current' in feature['name']:
            holding_current = feature['value']
            print(feature)
        elif 'soma.v.bpo_threshold_current' in feature['name']:
            threshold_current = feature['value']
            print(feature)

    return (holding_current, threshold_current)

def load_mechanism(directory_path):
    #Copy the mechanism in the working directory
    if os.path.exists("./x86_64") and os.path.isdir("./x86_64"):
        shutil.rmtree("./x86_64")
    source_folder = f"{directory_path}/x86_64/"
    destination_folder = "./x86_64/"
    shutil.copytree(source_folder, destination_folder)

def connect_forge(bucket, endpoint, access_token, forge_path=None):
    """Creation of a forge session"""

    forge = KnowledgeGraphForge(
        forge_path, bucket=bucket, endpoint=endpoint, token=access_token
    )
    return forge

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--emodel_id', action="store", type=str, required=True)
    args = parser.parse_args()
    emodel_id = args.emodel_id

    # Get metadata
    forge = connect_forge(bucket, endpoint, access_token, forge_path=forge_path)
    r = forge.retrieve(emodel_id)
    if r is None:
        raise ValueError(f"Resource with id {emodel_id} not found.")
    emodel = r.__dict__.get('eModel', r.__dict__.get('emodel'))
    etype = r.__dict__.get('eType', r.__dict__.get('etype', None))
    ttype = r.__dict__.get('tType', r.__dict__.get('ttype', None))
    mtype = r.__dict__.get('mType', r.__dict__.get('mtype', None))
    iteration_tag = r.__dict__.get("iteration", None)
    seed = r.__dict__.get("seed", None)
    description = r.__dict__.get("description", None)
    if description is not None:
        if "placeholder" in description:
            description = "placeholder"
        elif "detailed" in description:
            description = "detailed"

    if r.subject.species.label == "Rattus norvegicus":
        species = "rat"
    elif r.subject.species.label == "Mus musculus":
        species = "mouse"
    elif r.subject.species.label == "Homo sapiens":
        species = "human"
    else:
        raise ValueError(f"Species {r.subject.species.label} not supported.")

    brain_region = r.brainLocation.brainRegion.label

    metadata = {
        "emodel": emodel,
        "etype": etype,
        "mtype": mtype,
        "ttype": ttype,
        "species": species,
        "iteration_tag": iteration_tag,
        "brain_region": brain_region,
    }

    nap = NexusAccessPoint(
        **metadata,
        project=PROJECT,
        organisation=ORG,
        endpoint=endpoint,
        access_token=access_token,
        forge_path=forge_path,
        sleep_time=7,
    )

    # Download data from Nexus
    print("Downloading data...")
    model_configuration = nap.get_model_configuration()
    nap.get_hoc()
    nap.get_emodel()

    # Load the data and mechanism for simulation
    folder_id = nap.get_emodel().emodel_metadata.as_string()
    directory_path = f"./nexus_temp/{folder_id}"
    load_mechanism(directory_path=directory_path)
    hoc_file = Path(directory_path) / "model.hoc"
    morph_file = nap.download_morphology(model_configuration.morphology.name, model_configuration.morphology.format, model_configuration.morphology.id)
    holding_current, threshold_current = getHoldingThreshCurrent(directory_path)

    # Run the simulation
    from bluecellulab import Cell
    from bluecellulab import Simulation
    from bluecellulab.circuit.circuit_access import EmodelProperties
    from bluecellulab.simulation.neuron_globals import NeuronGlobals

    emodel_properties = EmodelProperties(threshold_current=threshold_current,
                                holding_current=holding_current)

    if threshold_current != 0:
        print("The emodel uses threshold based amplitudes")
        amplitudes = [x * threshold_current / 100 for x in amplitudes]

    fig, axes = plt.subplots(nrows=len(amplitudes), ncols=1, figsize=(10, 2*len(amplitudes)))
    fig.suptitle(nap.get_emodel().emodel_metadata.emodel, fontsize=16)
    print("Running simulation...")
    for i, amp in enumerate(amplitudes):
        cell = Cell(hoc_file, morph_file, template_format="v6", emodel_properties=emodel_properties)
        sim = Simulation()
        sim.add_cell(cell)
        cell.add_step(start_time=550.0, stop_time=950.0, level=amp) # step current injection
        NeuronGlobals.get_instance().temperature = TEMPERATURE
        NeuronGlobals.get_instance().v_init = V_INIT
        sim.run(1000, cvode=False, dt=0.025)
        time, voltage = cell.get_time(), cell.get_soma_voltage()
        if threshold_current != 0:
            axes[i].plot(time, voltage, label=f"step_{amplitudes[i]}")
        else:
            axes[i].plot(time, voltage, label=f"step_{amp}")
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Vm (mV)")
        axes[i].legend(loc='upper right')

    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.tight_layout()
    plt.savefig(f"./figures/{description}_{emodel}_{iteration_tag}_{seed}.png", dpi=300)
    print("Simulation completed. Results saved in ./figures/")


