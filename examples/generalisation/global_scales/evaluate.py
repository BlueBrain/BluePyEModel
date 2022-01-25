import pandas as pd
from bluepyemodel.access_point import get_access_point
from bluepyemodel.tools.misc_evaluators import feature_evaluation

if __name__ == "__main__":
    df = pd.read_csv("morphs_df.csv")
    df_synth = pd.read_csv("out/synthesis/synth_combos_df_cADpyr_L5TPC.csv")

    emodel = "cADpyr_L5TPC"
    emodel_db = get_access_point("local", emodel, emodel_dir="configs", legacy_dir_structure=True)
    df["emodel"] = emodel
    df["soma_model"] = df_synth.loc[0, "soma_model"]
    df["soma_scaler"] = 1.0
    df["AIS_model"] = df_synth.loc[0, "AIS_model"]
    df["AIS_scaler"] = 1.0

    df = feature_evaluation(
        df,
        emodel_db,
        morphology_path="morphology_path",
        parallel_factory="multiprocessing",
        score_threshold=20,
    )
    df.to_csv("data.csv")
