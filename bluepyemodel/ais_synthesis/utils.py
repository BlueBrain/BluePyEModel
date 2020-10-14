"""Util functions."""
import json
from copy import deepcopy
from functools import partial

import numpy as np

FEATURES_TO_REMOVE = ["bAP", "bpo_", "decay_time_constant_after_stim", "AP2_amp"]


def get_emodels(morphs_combos_df, emodels):
    """Convert emodel from 'all' to list of emodels."""
    if emodels == "all":
        return sorted(morphs_combos_df.emodel.unique())
    if not isinstance(emodels, list):
        return [emodels]
    return emodels


def get_mtypes(morphs_combos_df, mtypes):
    """Convert mtypes from 'all' to list of mtypes."""
    if mtypes == "all":
        return sorted(morphs_combos_df.mtype.unique())
    if not isinstance(mtypes, list):
        return [mtypes]
    return mtypes


def _filter_features(combo, features_to_remove=None):
    """delete features from list of unwanted features."""
    if features_to_remove is None:
        return combo
    if isinstance(combo["scores"], dict):
        keys = deepcopy(list(combo["scores"].keys()))
        for key in keys:
            for feat in features_to_remove:
                if feat in key.split("."):
                    del combo["scores"][key]
    return combo


def get_scores(morphs_combos_df, clip=250):
    """compute the median and max scores from computations on filtered features."""
    morphs_combos_df["scores_raw"] = morphs_combos_df["scores"]
    morphs_combos_df["scores"] = morphs_combos_df["scores_raw"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and len(s) > 0 else s
    )

    filter_features = partial(_filter_features, features_to_remove=FEATURES_TO_REMOVE)
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


def get_me_types_map(recipe_path, emodel_etype_path):
    """Use recipe data and bluepymm to get mtype/etype combos."""
    from bluepymm.prepare_combos.parse_files import read_mm_recipe

    recipe = read_mm_recipe(recipe_path)
    emodel_etype_map = json.load(open(emodel_etype_path, "rb"))
    for i, combos in recipe.iterrows():
        for emodel, emap in emodel_etype_map.items():
            if combos.layer in emap["layer"] and combos.etype == emap["etype"]:
                if "mtype" in emap:
                    if emap["mtype"] == combos.fullmtype:
                        recipe.loc[i, "emodel"] = emodel
                else:
                    recipe.loc[i, "emodel"] = emodel

    return recipe.rename(columns={"fullmtype": "mtype"})
