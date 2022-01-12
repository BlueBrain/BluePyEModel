import pandas as pd
from bluepyemodel.access_point import get_access_point
from bluepyemodel.tools.misc_evaluators import feature_evaluation

if __name__ == "__main__":
    df = pd.read_csv("morphs_df.csv")
    emodel = "cADpyr_L5TPC"
    emodel_db = get_access_point("local", emodel, emodel_dir="configs", legacy_dir_structure=True)
    df["emodel"] = emodel
    df = feature_evaluation(
        df,
        emodel_db,
        morphology_path="morphology_path",
        parallel_factory="multiprocessing",
        score_threshold=20,
    )
    df.to_csv("data.csv")
