import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from bluepyemodel.generalisation.utils import get_emodels
import neurom as nm

from bluepyemodel.generalisation.utils import get_scores


def get_surface_areas(df):
    for gid in df.index:
        m = nm.load_morphology(df.loc[gid, "morphology_path"])
        area = 0
        for neurite in m.neurites:
            if neurite.type != nm.NeuriteType.axon:
                area += neurite.area
        df.loc[gid, "area"] = area
    return df


def plot(df1, df2, threshold=5, pdf_filename="evaluations.pdf", feat_set=None):
    """Plot the results of ais synthesis evaluations."""
    emodels = get_emodels(df1, "all")
    df1["median_score"] = df1["median_score"].clip(0.0, 2 * threshold)
    df2["median_score"] = df2["median_score"].clip(0.0, 2 * threshold)
    df1 = get_surface_areas(df1)
    df2 = get_surface_areas(df2)
    for emodel in emodels:
        with PdfPages(pdf_filename) as pdf:

            mask = df1.emodel == emodel
            score_df1 = pd.DataFrame()
            score_df1["area"] = df1["area"]
            score_df2 = pd.DataFrame()
            score_df2["area"] = df2["area"]

            for score in json.loads(df1.loc[0, "scores_raw"]):
                score_df1[score] = df1["scores_raw"].apply(
                    lambda s, score=score: json.loads(s)[score]
                )
            for score in json.loads(df2.loc[0, "scores_raw"]):
                score_df2[score] = df2["scores_raw"].apply(
                    lambda s, score=score: json.loads(s)[score]
                )

            _df1 = score_df1[mask]
            __df1 = _df1.set_index("area")
            _df1 = _df1.drop(index=0).sort_values(by="area").set_index("area")

            _df2 = score_df2[mask]
            __df2 = _df2.set_index("area")
            _df2 = _df2.drop(index=0).sort_values(by="area").set_index("area")
            for feat in _df1.columns:
                if feat_set is None or (feat_set is not None and feat.startswith(feat_set)):
                    plt.figure(figsize=(4, 2))
                    clip = 5
                    plt.plot(
                        _df1.index,
                        np.clip(_df1[feat], 0, clip),
                        "-",
                        c="C0",
                        label="scaled adapted",
                    )
                    try:
                        plt.plot(
                            _df2.index, np.clip(_df2[feat], 0, clip), "-", c="C1", label="scaled"
                        )
                    except:
                        pass
                    plt.plot(
                        df1.area[0], np.clip(__df1.head(1)[feat], 0, clip), "or", label="exemplar"
                    )
                    try:
                        plt.plot(
                            df2.area[0],
                            np.clip(__df2.head(1)[feat], 0, clip),
                            "or",
                            label="exemplar",
                        )
                    except:
                        pass

                    plt.suptitle(feat)
                    plt.legend()
                    plt.gca().set_ylim(0, clip + 0.5)
                    plt.xlabel("surface area")
                    plt.ylabel("score")
                    pdf.savefig(bbox_inches="tight")
                    plt.close()


if __name__ == "__main__":
    df_scaled = pd.read_csv("out/evaluations/synth_combos_with_scores_df_cADpyr_L5TPC.csv")
    df_orig = pd.read_csv("data.csv")
    df_orig = get_scores(df_orig)
    # plot(df_scaled, df_orig, feat_set="Step_200", pdf_filename="evaluations_step_200.pdf")
    plot(df_scaled, df_orig, pdf_filename="evaluations.pdf")
