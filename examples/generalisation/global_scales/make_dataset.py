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


def add_global_scales(df):
    scales = np.linspace(0.5, 1.5, 70)

    i = 1
    for scale in scales:
        path = f"morphologies/morph_{scale}.asc"
        m = Morphology(df.loc[0, "morphology_path"])
        for root_section in m.root_sections[1:]:
            for section in root_section.iter():
                section.diameters *= scale
        m.write(path)
        df.loc[i, "scale"] = scale
        df.loc[i, "morphology_path"] = path
        i += 1
    return df


def fix_ais(df):
    m = Morphology(df.loc[0, "morphology_path"])
    m.root_sections[0].diameters = len(m.root_sections[0].points) * [1.9]
    path = "morphologies/fix_ais.asc"
    m.write(path)
    df.loc[0, "morphology_path"] = path
    return df


if __name__ == "__main__":
    if not Path("morphologies").exists():
        Path("morphologies").mkdir()

    base = "/gpfs/bbp.cscs.ch/project/proj83/home/gevaert/morph-release/morph_release_old_code-2020-07-27/output/04_ZeroDiameterFix-asc/"
    df = MorphDB.from_neurondb(base + "neuronDB.xml", morphology_folder=base).df[
        ["name", "path", "mtype"]
    ]
    df = df[df.name == "C060114A5"].reset_index(drop=True)
    df = df.rename(columns={"path": "morphology_path"})

    # make a copy with constant ais diameter
    df = fix_ais(df)

    # create morphologies with scaled diameters
    df = add_global_scales(df)

    df["mtype"] = df.loc[0, "mtype"]
    df["use_axon"] = True
    df.loc[1:, "name"] = df.loc[0, "name"] + "_scaled"
    print(df)

    df.to_csv("morphs_df.csv", index=False)
