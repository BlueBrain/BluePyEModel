"""ComboDB class to manage combos."""
import logging

import pandas as pd
import yaml
from morph_tool.morphdb import MorphDB

logger = logging.getLogger(__name__)


def _get_metypes(cell_composition):
    """From cell_composition dict file, create list of possible metypes."""
    if cell_composition["version"] not in ("v2.0",):
        raise Exception("Only v2.0 of recipe yaml files are supported")

    mecombos = pd.DataFrame(columns=["layer", "mtype", "etype"])
    for region in cell_composition["neurons"]:
        for etype in region["traits"]["etype"].keys():
            end = len(mecombos)
            mecombos.loc[end, "layer"] = str(region["traits"]["layer"])
            mecombos.loc[end, "mtype"] = str(region["traits"]["mtype"])
            mecombos.loc[end, "etype"] = str(etype)
    return mecombos


def _create_combos_df(df, me_models):
    """From the morphs_df, create a dataframe with all possible combos."""
    _dfs = []
    for combo_id in me_models.index:
        combo = df[df.mtype == me_models.loc[combo_id, "mtype"]]
        combo = combo.assign(
            etype=me_models.loc[combo_id, "etype"], emodel=me_models.loc[combo_id, "emodel"]
        )
        _dfs.append(combo.copy())
    combo_df = pd.concat(_dfs)

    return combo_df.drop_duplicates().reset_index()


class ComboDB(MorphDB):
    """Database for me-combos, from morphology release, cell composition and emoodel database"""

    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)
        self.combo_df = None

    @classmethod
    def from_neurondb(
        cls,
        neurondb,
        label="default",
        morphology_folder=None,
        cell_composition=None,
        emodel_db=None,
        filter_pc=True,
    ):
        """Constructor from neurondb.

        Args:
            neurondb (str): path to neurondb
            label (str): label of dataset
            morphology_folder (str): path to morphologies
            cell_composition (str|dict): path or dict with cell_composition.yaml
            emodel_db (BluePyEmodel.api.DatabaseAPI): emodel database
            filter_pc (bool): only assing emodels with corresponding layers for cADpyr
        """
        obj = MorphDB.from_neurondb(neurondb, label=label, morphology_folder=morphology_folder)
        if cell_composition is not None and emodel_db is not None:
            obj.combo_df = cls._set_combo_df(
                obj, cell_composition=cell_composition, emodel_db=emodel_db, filter_pc=filter_pc
            )
        return obj

    @classmethod
    def from_dataframe(cls, df, cell_composition=None, emodel_db=None, filter_pc=True):
        """Constructor from dataframe.

        Args:
            neurondb (str): path to neurondb
            morphology_folder (str): path to morphologies
            cell_composition (str|dict): path or dict with cell_composition.yaml
            emodel_db (BluePyEmodel.api.DatabaseAPI): emodel database
            filter_pc (bool): only assing emodels with corresponding layers for cADpyr
        """
        obj = MorphDB()
        obj.df = df
        if cell_composition is not None and emodel_db is not None:
            obj.combo_df = cls._set_combo_df(
                obj, cell_composition=cell_composition, emodel_db=emodel_db, filter_pc=filter_pc
            )
        return obj

    def _set_combo_df(self, cell_composition, emodel_db, filter_pc=True):
        """Create combo_df.

        Args:
            cell_composition (str|dict): path or dict with cell_composition.yaml
            emodel_db (BluePyEmodel.api.DatabaseAPI): emodel database
            filter_pc (bool): only assing emodels with corresponding layers for cADpyr
        """
        if isinstance(cell_composition, str):
            with open(cell_composition, "r") as f:
                cell_composition = yaml.safe_load(f)

        me_types = _get_metypes(cell_composition).set_index("etype")
        emodel_etypes_map_dict = emodel_db.get_emodel_etype_map()
        emodel_etypes_map = pd.DataFrame()
        emodel_etypes_map["emodel"] = list(emodel_etypes_map_dict.keys())
        emodel_etypes_map["etype"] = list(emodel_etypes_map_dict.values())
        me_models = me_types.join(emodel_etypes_map.set_index("etype")).reset_index()

        if filter_pc:
            gids = []
            for gid in me_models[me_models.etype == "cADpyr"].index:
                if me_models.loc[gid, "mtype"][1] != me_models.loc[gid, "emodel"].split("_")[1][1]:
                    gids.append(gid)
            me_models = me_models.drop(gids).reset_index()

        return _create_combos_df(self.df, me_models)


def add_for_optimisation_flag(emodel_db, morphs_combos_df):
    """Add flag for_optimisation for exemplar cells."""
    morphs_combos_df["for_optimisation"] = False
    for emodel in morphs_combos_df.emodel.unique():
        emodel_db.emodel = "_".join(emodel.split("_")[:2])
        morphology = emodel_db.get_morphologies()["name"]

        if len(morphs_combos_df[morphs_combos_df["name"] == morphology]) == 0:
            raise Exception(f"Exemplar for {emodel} named {morphology} does not exist")

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
