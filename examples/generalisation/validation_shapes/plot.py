import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bluepyemodel.generalisation.utils import get_scores

if __name__ == "__main__":
    df = pd.read_csv("data_exemplar.csv")
    df = get_scores(df)

    score_diffs = []
    for feat in df.loc[0, "scores"]:
        score_diffs.append(abs(df.loc[0, "scores"][feat] - df.loc[1, "scores"][feat]))
    plt.figure(figsize=(5, 3))
    plt.hist(score_diffs, bins=20)
    plt.xlabel("abs score differences")
    plt.savefig("soma_shape.pdf", bbox_inches="tight")

    plt.figure(figsize=(5, 3))
    all_scores = []
    for feat in df.loc[0, "scores"]:
        scores = []
        for gid in df.index[2:]:
            scores.append(abs(df.loc[gid, "scores"][feat] - df.loc[0, "scores"][feat]))
        all_scores.append(np.array(scores))
        plt.plot(df.AIS_scaler[2:], scores, "k", lw=0.2)
    plt.plot(df.AIS_scaler[2:], scores, "k", lw=0.2, label="single feature score")
    all_scores = np.array(all_scores).mean(0)
    plt.plot(df.AIS_scaler[2:], all_scores, c="r", label="mean scores")
    plt.axvline(df.loc[2 + np.argmin(all_scores), "AIS_scaler"], c="b", label="best score")
    plt.xlabel("AIS diameter")
    plt.ylabel("abs score differences")
    plt.legend(loc="best")
    print("scale", df.loc[2 + np.argmin(scores), "AIS_scaler"])
    plt.savefig("ais_shape.pdf", bbox_inches="tight")
