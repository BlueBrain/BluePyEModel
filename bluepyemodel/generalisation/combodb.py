"""ComboDB class to manage combos."""
import pandas as pd
import yaml
from morph_tool.morphdb import MorphDB


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

    # to remove when MorphDB is updated
    combo_df["axon_inputs"] = combo_df["axon_inputs"].apply(tuple)

    return combo_df.drop_duplicates().reset_index()


class ComboDB(MorphDB):
    """Database for me-combos, from morphology release, cell composition and emoodel database"""

    def __init__(self, *args, **kwargs):
        """"""
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
    ):
        obj = MorphDB.from_neurondb(neurondb, label=label, morphology_folder=morphology_folder)
        if cell_composition is not None and emodel_db is not None:
            obj.combo_df = cls._set_combo_df(
                obj, cell_composition=cell_composition, emodel_db=emodel_db
            )
        return obj

    def _set_combo_df(self, cell_composition, emodel_db):
        """Create combo_df."""

        with open(cell_composition, "r") as f:
            cell_composition = yaml.safe_load(f)

        me_types = _get_metypes(cell_composition).set_index("etype")
        emodel_etypes_map_dict = emodel_db.get_emodel_etype_map()
        emodel_etypes_map = pd.DataFrame()
        emodel_etypes_map["emodel"] = list(emodel_etypes_map_dict.keys())
        emodel_etypes_map["etype"] = list(emodel_etypes_map_dict.values())
        me_models = me_types.join(emodel_etypes_map.set_index("etype")).reset_index()
        return _create_combos_df(self.df, me_models)
