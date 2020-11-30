"""Task to create combos from morphologies dataframe."""
import json
import logging

import pandas as pd
import yaml

import luigi

from .base_task import BaseTask
from .config import MorphComboLocalTarget
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def apply_substitutions(original_morphs_df, substitution_rules=None):
    """Applies substitution rule on .dat file.

    Args:
        original_morphs_df (DataFrame): dataframe with morphologies
        substitution_rules (dict): rules to assign duplicated mtypes to morphologies

    Returns:
        DataFrame: dataframe with original and new morphologies
    """
    if not substitution_rules:
        return original_morphs_df

    new_morphs_df = original_morphs_df.copy()
    for gid in original_morphs_df.index:
        mtype_orig = original_morphs_df.loc[gid, "mtype"]
        if mtype_orig in substitution_rules:
            for mtype in substitution_rules[mtype_orig]:
                new_cell = original_morphs_df.loc[gid].copy()
                new_cell["mtype"] = mtype
                new_morphs_df = new_morphs_df.append(new_cell)
    return new_morphs_df


def _add_for_optimisation_flag(emodel_db, morphs_combos_df):
    """Add flag for_optimisation for exemplar cells."""
    morphs_combos_df["for_optimisation"] = False
    for emodel in morphs_combos_df.emodel.unique():
        morphology = emodel_db.get_morphologies(emodel)[0]["name"]
        mask = (morphs_combos_df["name"] == morphology) & (morphs_combos_df.emodel == emodel)
        if len(morphs_combos_df[mask]) == 0:
            new_row = morphs_combos_df[morphs_combos_df["name"] == morphology].iloc[0]
            new_row["emodel"] = emodel
            new_row["for_optimisation"] = True
            etype = morphs_combos_df[morphs_combos_df.emodel == emodel].etype.unique()[0]
            new_row["etype"] = etype
            logger.warning("Emodel %s has a cell from a non-compatible etype %s", emodel, etype)
            morphs_combos_df = morphs_combos_df.append(new_row.copy())
        else:
            morphs_combos_df.loc[mask, "for_optimisation"] = True
    return morphs_combos_df.reset_index(drop=True)


def _get_me_types_map(recipe, emodel_etype_map):
    """Use recipe data and bluepymm to get mtype/etype combos."""
    me_types_map = pd.DataFrame()
    for i in recipe.index:
        combo = recipe.loc[i]
        for emodel, emap in emodel_etype_map.items():
            if combo.layer in emap["layer"] and combo.etype == emap["etype"]:
                if "mtype" in emap:
                    if emap["mtype"] == combo.fullmtype:
                        combo["emodel"] = emodel
                        me_types_map = me_types_map.append(combo.copy())
                else:
                    combo["emodel"] = emodel
                    me_types_map = me_types_map.append(combo.copy())

    return me_types_map.rename(columns={"fullmtype": "mtype"}).reset_index(drop=True)


def _get_mecombos(cell_composition):
    """From cell_composition dict file, create list of possible mecombos."""
    if cell_composition["version"] not in ("v2.0",):
        raise Exception("Only v2.0 of recipe yaml files are supported")

    mecombos = pd.DataFrame(columns=["layer", "fullmtype", "etype"])
    for region in cell_composition["neurons"]:
        for etype in region["traits"]["etype"].keys():
            end = len(mecombos)
            mecombos.loc[end, "layer"] = str(region["traits"]["layer"])
            mecombos.loc[end, "fullmtype"] = str(region["traits"]["mtype"])
            mecombos.loc[end, "etype"] = str(etype)
    return mecombos


def _filter_me_types_map(orig_me_types_map, full_emodels):
    """Filters emodel map with full_emodels (including seeds) and list of wnated emodels."""
    _dfs = []
    for full_emodel, emodel in full_emodels.items():
        _df = orig_me_types_map[orig_me_types_map.emodel == emodel]
        _df = _df.assign(emodel=full_emodel)
        _dfs.append(_df)
    return pd.concat(_dfs).reset_index(drop=True)


def _create_morphs_combos_df(morphs_df, me_types_map):
    """From the morphs_df, create a dataframe with all possible combos."""
    morphs_combos_df = pd.DataFrame()
    for combo_id in me_types_map.index:
        combo = morphs_df[morphs_df.mtype == me_types_map.loc[combo_id, "mtype"]]
        combo = combo.assign(etype=me_types_map.loc[combo_id, "etype"])
        combo = combo.assign(emodel=me_types_map.loc[combo_id, "emodel"])
        morphs_combos_df = morphs_combos_df.append(combo.copy())

    morphs_combos_df = (
        morphs_combos_df.drop_duplicates().reset_index().rename(columns={"index": "morph_gid"})
    )
    return morphs_combos_df


class ApplySubstitutionRules(BaseTask):
    """Apply substitution rules to the morphology dataframe.

    Args:
        substitution_rules (dict): rules to assign duplicated mtypes to morphologies
    """

    morphs_df_path = luigi.Parameter(default="morphs_df.csv")
    substitution_rules_path = luigi.Parameter(default="substitution_rules.yaml")
    target_path = luigi.Parameter(default="substituted_morphs_df.csv")

    def run(self):
        """"""
        with open(self.substitution_rules_path, "rb") as sub_file:
            substitution_rules = yaml.full_load(sub_file)

        substituted_morphs_df = apply_substitutions(
            pd.read_csv(self.morphs_df_path), substitution_rules
        )
        ensure_dir(self.output().path)
        substituted_morphs_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return MorphComboLocalTarget(self.target_path)


class CreateMorphCombosDF(BaseTask):
    """Create a dataframe with all combos to run."""

    cell_composition_path = luigi.Parameter()
    emodel_etype_map_path = luigi.Parameter()
    target_path = luigi.Parameter(default="morphs_combos_df.csv")

    def requires(self):
        """"""
        return ApplySubstitutionRules()

    def run(self):
        """"""
        morphs_df = pd.read_csv(self.input().path)
        cell_composition = yaml.safe_load(open(self.cell_composition_path, "r"))
        emodel_etype_map = json.load(open(self.emodel_etype_map_path, "rb"))

        mecombos = _get_mecombos(cell_composition)
        me_types_map = _get_me_types_map(mecombos, emodel_etype_map)

        me_types_map = _filter_me_types_map(me_types_map, self.emodel_db.get_emodel_names())
        morphs_combos_df = _create_morphs_combos_df(morphs_df, me_types_map)
        morphs_combos_df = _add_for_optimisation_flag(self.emodel_db, morphs_combos_df)

        ensure_dir(self.output().path)
        morphs_combos_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return MorphComboLocalTarget(self.target_path)
