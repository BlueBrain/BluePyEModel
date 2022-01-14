import pandas as pd
from pathlib import Path
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


if __name__ == "__main__":

    #base = "/gpfs/bbp.cscs.ch/project/proj83/home/gevaert/morph-release/morph_release_old_code-2020-07-27/output/04_ZeroDiameterFix-asc/"
    base = "/gpfs/bbp.cscs.ch/project/proj83/home/gevaert/morph-release/morph_release_old_code-2020-07-27/output/06_RepairUnravel-asc/"
    df = MorphDB.from_neurondb(base + "neuronDB.xml", morphology_folder=base).df[
        ["name", "path", "mtype", "use_axon"]
    ]
    df = df[df.mtype == "L5_TPC:A"].reset_index(drop=True)
    df = df.rename(columns={"path": "morphology_path"})

    print(df)

    df.to_csv("morphs_df.csv", index=False)
