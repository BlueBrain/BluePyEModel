"""Util functions."""
import json
from copy import deepcopy
from functools import partial

import numpy as np


def get_emodels(morphs_combos_df, emodels):
    """Convert emodel from 'all' or None to list of emodels."""
    if emodels == "all" or emodels is None:
        return sorted(morphs_combos_df.emodel.unique())
    if not isinstance(emodels, list):
        return [emodels]
    return emodels


def get_mtypes(morphs_combos_df, mtypes):
    """Convert mtypes from 'all' or None to list of mtypes."""
    if mtypes == "all" or mtypes is None:
        return sorted(morphs_combos_df.mtype.unique())
    if not isinstance(mtypes, list):
        return [mtypes]
    return mtypes


def _filter_features(combo, features=None, method="ignore"):
    """delete or keep features if method==ignore, resp. method==keep."""
    if isinstance(combo["scores"], dict):
        keys = deepcopy(list(combo["scores"].keys()))
        for key in keys:
            if method == "ignore":
                for feat in features:
                    if feat in key.split("."):
                        del combo["scores"][key]
            elif method == "keep":
                if not any([feat in key.split(".") for feat in features]):
                    if key in combo["scores"]:
                        del combo["scores"][key]
    return combo


def get_scores(morphs_combos_df, features_to_ignore=None, features_to_keep=None, clip=250):
    """compute the median and max scores from computations on filtered features."""
    morphs_combos_df["scores_raw"] = morphs_combos_df["scores"]
    morphs_combos_df["scores"] = morphs_combos_df["scores_raw"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and len(s) > 0 else s
    )

    if features_to_ignore is not None:
        if features_to_keep is not None:
            raise Exception("please provide only a list of features to ignore or to keep")
        filter_features = partial(_filter_features, features=features_to_ignore, method="ignore")

    if features_to_keep is not None:
        if features_to_ignore is not None:
            raise Exception("please provide only a list of features to ignore or to keep")
        filter_features = partial(_filter_features, features=features_to_keep, method="keep")

    morphs_combos_df.apply(filter_features, axis=1)
    morphs_combos_df["median_score"] = morphs_combos_df["scores"].apply(
        lambda score: np.clip(np.median(list(score.values())), 0, clip)
        if isinstance(score, dict)
        else np.nan
    )
    morphs_combos_df["max_score"] = morphs_combos_df["scores"].apply(
        lambda score: np.clip(np.max(list(score.values())), 0, clip)
        if isinstance(score, dict)
        else np.nan
    )

    return morphs_combos_df
