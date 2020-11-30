"""Luigi task for select step with plotting tasks."""
import logging

import pandas as pd
import yaml

import luigi

from ...ais_synthesis.select import apply_megating, select_best_emodel
from .base_task import BaseTask
from .config import SelectConfig, SelectLocalTarget
from .gather import GatherGenericEvaluations, GatherSynthEvaluations, GatherExemplarEvaluations
from .utils import ensure_dir

logger = logging.getLogger(__name__)


class ApplyGenericMegating(BaseTask):
    """Apply megating to scores."""

    emodels = luigi.ListParameter()
    base_threshold = luigi.FloatParameter(default=5)
    no_threshold = luigi.FloatParameter(default=250)
    target_path = luigi.Parameter(default="megated_scores_df.csv")

    def requires(self):
        """"""
        return {
            "exemplar": GatherExemplarEvaluations(emodels=self.emodels),
            "generic": GatherGenericEvaluations(emodels=self.emodels),
        }

    def run(self):
        """"""
        selected_combos_df = pd.read_csv(self.input()["generic"].path)
        exemplar_combos_df = pd.read_csv(self.input()["exemplar"].path)
        megate_thresholds = yaml.full_load(open(SelectConfig().megate_thresholds_path, "r"))

        megated_scores_df = apply_megating(
            selected_combos_df,
            exemplar_combos_df,
            megate_thresholds,
            self.base_threshold,
            self.no_threshold,
        )
        megated_scores_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return SelectLocalTarget(self.target_path)


class ApplyMegating(BaseTask):
    """Apply megating to scores."""

    emodels = luigi.ListParameter()
    base_threshold = luigi.FloatParameter(default=5)
    no_threshold = luigi.FloatParameter(default=250)
    target_path = luigi.Parameter(default="megated_scores_df.csv")

    def requires(self):
        """"""
        return {
            "exemplar": GatherExemplarEvaluations(emodels=self.emodels),
            "synth": GatherSynthEvaluations(emodels=self.emodels),
        }

    def run(self):
        """"""
        selected_combos_df = pd.read_csv(self.input()["synth"].path)
        exemplar_combos_df = pd.read_csv(self.input()["exemplar"].path)
        megate_thresholds = yaml.full_load(open(SelectConfig().megate_thresholds_path, "r"))
        megated_scores_df = apply_megating(
            selected_combos_df,
            exemplar_combos_df,
            megate_thresholds,
            self.base_threshold,
            self.no_threshold,
        )
        ensure_dir(self.output().path)
        megated_scores_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return SelectLocalTarget(self.target_path)


class SelectCombos(BaseTask):
    """Select combos with  scores under a threshold."""

    emodels = luigi.ListParameter()
    best_emodel_path = luigi.Parameter(default="best_emodels.csv")
    target_path = luigi.Parameter(default="selected_combos_df.csv")

    def requires(self):
        """"""
        return {
            "megated": ApplyMegating(emodels=self.emodels),
            "synth": GatherSynthEvaluations(emodels=self.emodels),
        }

    def run(self):
        """"""
        selected_combos_df = pd.read_csv(self.input()["synth"].path)
        megated_scores_df = pd.read_csv(self.input()["megated"].path)

        selected_combos_df["selected"] = megated_scores_df.all(axis=1)
        # pylint: disable=protected-access
        select_best_emodel(selected_combos_df, self.output()._prefix / self.best_emodel_path)
        ensure_dir(self.output().path)
        selected_combos_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return SelectLocalTarget(self.target_path)


class SelectGenericCombos(BaseTask):
    """Select combos with  scores under a threshold."""

    emodels = luigi.ListParameter()
    best_emodel_path = luigi.Parameter(default="best_emodels.csv")

    def requires(self):
        """"""
        return {
            "megated": ApplyGenericMegating(emodels=self.emodels),
            "generic": GatherGenericEvaluations(emodels=self.emodels),
        }

    def run(self):
        """"""
        selected_combos_df = pd.read_csv(self.input()["generic"].path)
        megated_scores_df = pd.read_csv(self.input()["megated"].path)
        selected_combos_df["selected"] = megated_scores_df.all(axis=1)

        select_best_emodel(selected_combos_df, self.best_emodel_path)
        ensure_dir(self.output().path)
        selected_combos_df.to_csv(self.output().path, index=False)

    def output(self):
        """"""
        return SelectLocalTarget(self.target_path)
