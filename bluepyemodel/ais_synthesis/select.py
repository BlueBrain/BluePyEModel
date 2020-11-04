"""Function to apply select step to evaluated combos."""
import json
import re
import pandas as pd
from tqdm import tqdm


def select_best_emodel(selected_combos_df, best_emodel_path):
    """From all possible emodels of an me-type, choose the one with  highest fraction of pass."""
    df = (
        selected_combos_df[["mtype", "emodel", "etype", "selected"]]
        .groupby(["mtype", "etype", "emodel"])
        .mean()
    )
    best_df = df.reset_index().groupby(["mtype", "etype"]).idxmax()
    best_df["emodel"] = best_df["selected"].apply(
        lambda index: df.reset_index().loc[index, "emodel"]
    )
    best_df.drop("selected", axis=1).reset_index().to_csv(best_emodel_path, index=False)


def _get_score_df(df, megate_thresholds):
    """Create df with scores only and remove some features."""
    score_df = df["scores_raw"].apply(
        lambda json_str: pd.Series(json.loads(json_str))
        if isinstance(json_str, str)
        else pd.Series(dtype=float)
    )
    for feat_to_remove in megate_thresholds["ignore"]:
        score_df = score_df.filter(regex="^((?!" + feat_to_remove + ").)*$")

    return score_df


def _in_threshold(mtype, emodel, threshold):
    """Some logic to detect if a me-combos is in threshold dict."""
    if "mtype" in threshold and mtype not in threshold["mtype"]:
        return False

    if "emodel" in threshold and emodel not in threshold["emodel"]:
        return False

    return True


def _get_threshold(mtype, emodel, col, megate_thresholds, base_threshold=5, no_threshold=250):
    """Get threshold  values, with hardcoded rules from MM setup."""
    if "thresholds" not in megate_thresholds or megate_thresholds["thresholds"] is None:
        return base_threshold

    for megate_threshold in megate_thresholds["thresholds"]:
        if _in_threshold(mtype, emodel, megate_threshold):
            for feat in megate_threshold["features"]:
                feat_reg = re.compile(feat)
                if feat_reg.match(col):
                    return no_threshold

    return base_threshold


def apply_megating(
    repaired_df,
    repaired_exemplar_df,
    megate_thresholds,
    base_threshold=5,
    no_threshold=250,
):
    """Apply me-gating a la BluePyMM.

    For each cell, each scores are compared with the scores of the exemplar cell. If the score is
    smaller  than max(threshold, threshold * exemplar_score),  then the cell is flagged as pass.
    The value of threshold is base_treshold for all features, except for the ones provided by
    megate_thresholds, where it is no_threshold.
    """
    # fillna is to ensures no score will pass in the inequality below
    repaired_score_df = _get_score_df(repaired_df, megate_thresholds).fillna(0.0)
    repaired_exemplar_score_df = _get_score_df(repaired_exemplar_df, megate_thresholds).fillna(
        250.0
    )
    for emodel in tqdm(repaired_exemplar_df.emodel.unique()):
        exemplar_id = repaired_exemplar_df[
            (repaired_exemplar_df.emodel == emodel) & (repaired_exemplar_df.for_optimisation == 1)
        ].index

        if len(exemplar_id) > 1:
            raise Exception("Found more than one exemplar, something is wrong.")
        exemplar_id = exemplar_id[0]

        for mtype in repaired_df[(repaired_df.emodel == emodel)].mtype.unique():
            ids = repaired_df[(repaired_df.emodel == emodel) & (repaired_df.mtype == mtype)].index
            for col in repaired_exemplar_score_df.columns:
                threshold = _get_threshold(
                    mtype,
                    emodel,
                    col,
                    megate_thresholds,
                    base_threshold=base_threshold,
                    no_threshold=no_threshold,
                )
                repaired_score_df.loc[ids, col] = repaired_score_df.loc[ids, col] <= max(
                    threshold,
                    threshold * repaired_exemplar_score_df.loc[exemplar_id, col],
                )
    return repaired_score_df
