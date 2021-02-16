"""Task to create combos from morphologies dataframe."""
import luigi
import pandas as pd
import yaml

from bluepyemodel.generalisation.combodb import ComboDB
from bluepyemodel.generalisation.combodb import add_for_optimisation_flag
from bluepyemodel.tasks.generalisation.base_task import BaseTask
from bluepyemodel.tasks.generalisation.config import MorphComboLocalTarget
from bluepyemodel.tasks.generalisation.utils import ensure_dir


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
        morphs_combos_df = ComboDB.from_dataframe(
            morphs_df, cell_composition, self.emodel_db
        ).combo_df
        morphs_combos_df = add_for_optimisation_flag(self.emodel_db, morphs_combos_df)

        ensure_dir(self.output().path)
        morphs_combos_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return MorphComboLocalTarget(self.target_path)
