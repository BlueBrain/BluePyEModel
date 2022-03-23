import pandas as pd
import json
import matplotlib.pyplot as plt
from neurom import load_morphology, get
from neurom.core import Morphology as M
from morphio.mut import Morphology
from bluepyemodel.generalisation.evaluators import (
    evaluate_rho_axon,
    evaluate_combos_rho,
    evaluate_rho,
)
from bluepyemodel.access_point import get_access_point
from bluepyemodel.tools.misc_evaluators import feature_evaluation
from emodel_generalisation.utils import evaluate_rin
import numpy as np
from morph_tool.morphdb import MorphDB

from morph_tool.neuron_surface import get_NEURON_surface

if __name__ == "__main__":
    # get exemplar morpho
    base = "/gpfs/bbp.cscs.ch/project/proj83/home/gevaert/morph-release/morph_release_old_code-2020-07-27/output/04_ZeroDiameterFix-asc/"
    df = MorphDB.from_neurondb(base + "neuronDB.xml", morphology_folder=base).df[
        ["name", "path", "mtype"]
    ]
    df = df[df.name == "C060114A5"].reset_index(drop=True)

    m = Morphology(df.loc[0, "path"])

    # replace soma with model of same surface area
    area = get_NEURON_surface(str(df.loc[0, "path"]))
    df.loc[1, "soma_model"] = json.dumps(
        {"soma_surface": float(area), "soma_radius": float(get("soma_radius", M(m)))}
    )
    df.loc[1, "soma_scaler"] = 1
    df.loc[1, "name"] = df.loc[0, "name"] + "_soma_scaled"
    df.loc[1, "path"] = df.loc[0, "path"]

    # replace AIS with AIS of various sizes
    scales = np.linspace(1.8, 2.0, 30)
    for i, scale in enumerate(scales):
        df.loc[i+2, "AIS_model"] = json.dumps({'popt':[60, 0.0, 9.2, 1.0]})
        df.loc[i+2, "AIS_scaler"] = scale

    df["mtype"] = df.loc[0, "mtype"]
    df["use_axon"] = True
    df.loc[2:, "name"] = df.loc[0, "name"] + "_ais_scaled"
    df.loc[2:, "path"] = df.loc[0, "path"]
    print(df)

    # evaluate feature and scores
    emodel = "cADpyr_L5TPC"
    emodel_db = get_access_point("local", emodel, emodel_dir="configs", legacy_dir_structure=True)
    df["emodel"] = emodel
    df = feature_evaluation(
        df,
        emodel_db,
        morphology_path="path",
        parallel_factory="multiprocessing",
        score_threshold=20,
    )
    df.to_csv("data_exemplar.csv")
