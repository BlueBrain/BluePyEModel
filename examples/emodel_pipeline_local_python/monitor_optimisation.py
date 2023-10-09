"""
Copyright 2023, EPFL/Blue Brain Project

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

def monitor_optimisation():
    githash = "YOUR_GITHASH_HERE"
    emodel = "L5PC"
    print("emodel: ", emodel)
    print("Githash: ", githash)

    best_fitness = []
    if githash is None:
        paths = glob.glob(f"./checkpoints/{emodel}/*.pkl", recursive=True)
    else:
        paths = glob.glob(f"./checkpoints/{emodel}/{githash}/*.pkl", recursive=True)

    if not paths:
        print("No checkpoints found.")

    for path in paths:
        data = pickle.load(open(path, "rb"))
        seed = path.split("_seed=")[1].split(".pkl")[0]
        generation = data["logbook"].select("gen")[-1]
        best_score = sum(data["halloffame"][0].fitness.values)
        print(f"Seed: {seed}, Generation: {generation}, Score: {best_score}")
        best_fitness.append(sum(data["halloffame"][0].fitness.values))

    if best_fitness:
        print(f"Best fitness: {min(best_fitness)} from checkpoint {paths[numpy.argmin(best_fitness)]}")

if __name__ == '__main__':
    monitor_optimisation()