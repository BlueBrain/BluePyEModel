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