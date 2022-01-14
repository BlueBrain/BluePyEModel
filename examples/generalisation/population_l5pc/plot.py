import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import neurom as nm

from multiprocessing.pool import Pool
from matplotlib.colors import Normalize
from functools import partial


if __name__ == "__main__":

    clip = 3
    df_orig = pd.read_csv("out/evaluations/generic_combos_with_scores_df_cADpyr_L5TPC.csv")
    df_orig["median_score"] = np.clip(df_orig.median_score, 0, clip)

    df = pd.read_csv("out/evaluations/synth_combos_with_scores_df_cADpyr_L5TPC.csv")
    df["median_score"] = np.clip(df.median_score, 0, clip)

    plt.figure(figsize=(5, 3))
    plt.scatter(df_orig.median_score, df.median_score)
    plt.plot([0, clip], [0, clip])
    plt.axis([0, clip, 0, clip])
    plt.xlabel("original median scores")
    plt.ylabel("adapted median scores")
    plt.tight_layout()
    plt.savefig("score_comparison.pdf")

