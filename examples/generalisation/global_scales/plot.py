import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from bluepyemodel.generalisation.utils import get_emodels

from bluepyemodel.generalisation.utils import get_scores

def plot(df1, df2, threshold=5, pdf_filename="evaluations.pdf", feat_set=None):
    """Plot the results of ais synthesis evaluations."""
    emodels = get_emodels(df1, "all")
    df1["median_score"] = df1["median_score"].clip(0.0, 2 * threshold)
    df2["median_score"] = df2["median_score"].clip(0.0, 2 * threshold)

    for emodel in emodels:
        with PdfPages(pdf_filename) as pdf:

            mask = df1.emodel == emodel
            score_df1 = pd.DataFrame()
            score_df1["rho_axon"] = df1["scale"]
            score_df2 = pd.DataFrame()
            score_df2["rho_axon"] = df2["scale"]

            for score in json.loads(df1.loc[0, "scores_raw"]):
                score_df1[score] = df1["scores_raw"].apply(
                    lambda s, score=score: json.loads(s)[score]
                )
            for score in json.loads(df2.loc[0, "scores_raw"]):
                score_df2[score] = df2["scores_raw"].apply(
                    lambda s, score=score: json.loads(s)[score]
                )

            _df1 = score_df1[mask]
            _df1 = _df1.set_index("rho_axon")

            _df2 = score_df2[mask]
            _df2 = _df2.set_index("rho_axon")
            for feat in _df1.columns:
                if feat_set is None or (feat_set is not None and feat.startswith(feat_set)):
                    plt.figure(figsize=(4, 2))
                    clip = 5
                    plt.plot(_df1.index, np.clip(_df1[feat], 0, clip), "-", c='C0')
                    plt.plot(_df2.index, np.clip(_df2[feat], 0, clip), "-", c='C1')
                    plt.plot(1.0, np.clip(_df1.head(1)[feat], 0, clip), "or")
                    plt.suptitle(feat)
                    plt.gca().set_ylim(0, clip + 0.5)
                    plt.xlabel('diameter scale')
                    plt.ylabel('score')
                    pdf.savefig(bbox_inches='tight')
                    plt.close()


if __name__ == "__main__":
    df_scaled = pd.read_csv("out/evaluations/synth_combos_with_scores_df_cADpyr_L5TPC.csv")
    df_orig = pd.read_csv("data.csv")
    df_orig = get_scores(df_orig)
    #plot(df_scaled, df_orig, feat_set='Step_200', pdf_filename='evaluations_step_200.pdf')
    plot(df_scaled, df_orig, pdf_filename='evaluations_new.pdf')
