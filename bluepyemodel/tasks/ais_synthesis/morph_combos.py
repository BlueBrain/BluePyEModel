"""Task to create combos from morphologies dataframe."""
import json

import pandas as pd
import yaml

import luigi

from .base_task import BaseTask
from .utils import ensure_dir


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


def _filter_me_types_map(orig_me_types_map, full_emodels, all_emodels):
    """Filters emodel map with full_emodels (including seeds) and list of wnated emodels."""
    _dfs = []
    for full_emodel in all_emodels:
        _df = orig_me_types_map[orig_me_types_map.emodel == full_emodels[full_emodel]]
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


class CreateMorphCombosDF(BaseTask):
    """Create a dataframe with all combos to run."""

    morphs_df_path = luigi.Parameter(default="morphs_df.csv")
    cell_composition_path = luigi.Parameter()
    emodel_etype_map_path = luigi.Parameter()
    emodels = luigi.ListParameter(default=None)

    def run(self):
        """"""
        morphs_df = pd.read_csv(self.morphs_df_path)
        cell_composition = yaml.safe_load(open(self.cell_composition_path, "r"))
        emodel_etype_map = json.load(open(self.emodel_etype_map_path, "rb"))

        mecombos = _get_mecombos(cell_composition)
        me_types_map = _get_me_types_map(mecombos, emodel_etype_map)

        full_emodels = self.get_database().get_emodel_names()
        if self.emodels is not None:
            all_emodels = [
                full_emodel
                for full_emodel, emodel in full_emodels.items()
                if emodel in self.emodels  # pylint: disable=unsupported-membership-test
            ]
        else:
            all_emodels = list(full_emodels.keys())
        me_types_map = _filter_me_types_map(me_types_map, full_emodels, all_emodels)
        morphs_combos_df = _create_morphs_combos_df(morphs_df, me_types_map)

        ensure_dir(self.output().path)
        morphs_combos_df.to_csv(self.output().path)
