"""
Copyright 2023-2024 Blue Brain Project / EPFL

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

import pickle
import glob
import numpy
from bluepyemodel.access_point.access_point import OptimisationState
from bluepyemodel.access_point.local import LocalAccessPoint


def monitor_optimisation():

    # IMPORTANT: Please populate the following parameters with the relevant values utilized:
    githash = "YOUR_GITHASH_HERE"
    emodel = "L5PC"
    species = "rat"
    brain_region = "SSCX"

    print("emodel: ", emodel)
    print("Githash: ", githash)
    print("species: ", species)
    print("brain_region: ", brain_region)

    if githash is None:
        emodel_dir = "."
    else:
        emodel_dir = f"./run/{githash}"

    access_point = LocalAccessPoint(
        emodel=emodel,
        final_path="./final.json",
        species=species,
        brain_region=brain_region,
        emodel_dir=emodel_dir,
        iteration_tag=githash,
        recipes_path="./config/recipes.json",
    )

    best_fitness = []
    if githash is None:
        paths = glob.glob(f"./checkpoints/{emodel}/*.pkl", recursive=True)
    else:
        paths = glob.glob(f"./checkpoints/{emodel}/{githash}/*.pkl", recursive=True)

    if not paths:
        print("No checkpoints found.")

    for path in paths:
        with open(path, "rb") as file:
            data = pickle.load(file)
        seed = path.split("_seed=")[1].split(".pkl")[0]
        generation = data["logbook"].select("gen")[-1]
        best_score = sum(data["halloffame"][0].fitness.values)
        opt_state = access_point.optimisation_state(seed, continue_opt=True)
        if opt_state == OptimisationState.COMPLETED:
            status = "completed"
        elif opt_state == OptimisationState.IN_PROGRESS:
            status = "in progress"
        elif opt_state == OptimisationState.EMPTY:
            print(f"No checkpoint found for species: {species}, brain_region: {brain_region}")
            continue
        else:
            status = "unknown"
        print(f"Seed: {seed}, Generation: {generation}, Status: {status}, Score: {best_score}")
        best_fitness.append(best_score)

    if best_fitness:
        print(
            f"Best fitness: {min(best_fitness)} from checkpoint {paths[numpy.argmin(best_fitness)]}"
        )


if __name__ == "__main__":
    monitor_optimisation()
