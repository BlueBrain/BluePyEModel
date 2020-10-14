"""Luigi task for select step with plotting tasks."""
import json
import logging
import re

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import luigi

from .gather import GatherGenericEvaluations, GatherSynthEvaluations
from ...ais_synthesis.evaluators import evaluate_combos_rho_scores
from ...ais_synthesis.plotting import (
    plot_feature_select,
    plot_feature_summary_select,
    plot_frac_exceptions,
    plot_summary_select,
)
from ...ais_synthesis.utils import FEATURES_TO_REMOVE
from .base_task import BaseTask
from .utils import ensure_dir
from .config import selectconfigs

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def _select_best_emodel(selected_combos_df, best_emodel_path):
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


class EvaluateExemplars(BaseTask):
    """Evaluate all exemplars for select step with tresholds a la MM."""

    emodels = luigi.ListParameter(default=["all"])
    save_traces = luigi.BoolParameter(default=False)
    morphology_path = luigi.Parameter(default="morphology_path")
    eval_db_path = luigi.Parameter(default="eval_db.sql")

    def run(self):
        """Run."""
        morphs_combos_df = pd.read_csv(selectconfigs().exemplar_morphs_combos_df_path)
        morphs_combos_df = morphs_combos_df[morphs_combos_df.for_optimisation == 1]

        ensure_dir(self.eval_db_path)
        morphs_combos_with_scores_df = evaluate_combos_rho_scores(
            morphs_combos_df,
            self.emodel_db,
            emodels=self.emodels,
            morphology_path=self.morphology_path,
            continu=self.continu,
            ipyp_profile=self.ipyp_profile,
            save_traces=self.save_traces,
            combos_db_filename=self.eval_db_path,
        )

        ensure_dir(self.output().path)
        morphs_combos_with_scores_df.to_csv(self.output().path, index=False)


def _get_score_df(df):
    """Create df with scores only and remove some features."""
    score_df = df["scores_raw"].apply(
        lambda json_str: pd.Series(json.loads(json_str))
        if isinstance(json_str, str)
        else pd.Series(dtype=float)
    )
    for feat_to_remove in FEATURES_TO_REMOVE:
        score_df = score_df.filter(regex="^((?!" + feat_to_remove + ").)*$")

    return score_df


def _in_threshold(mtype, emodel, threshold):
    """Some logic to detect if a me-combos is in threshold dict."""
    to_use = False
    if "mtype" in threshold and mtype in threshold["mtype"]:
        to_use = True
    if "mtype" in threshold and mtype not in threshold["mtype"]:
        to_use = False

    if "emodel" in threshold and emodel in threshold["emodel"]:
        to_use = True
    if "emodel" in threshold and emodel not in threshold["emodel"]:
        to_use = False
    return to_use


def _get_threshold(
    mtype, emodel, col, megate_thresholds, base_threshold=5, no_threshold=250
):
    """Get threshold  values, with hardcoded rules from MM setup."""
    for megate_threshold in megate_thresholds:
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
    """Apply me-gating a la BluePyMM."""
    repaired_score_df = _get_score_df(repaired_df).fillna(True)
    repaired_exemplar_score_df = _get_score_df(repaired_exemplar_df).fillna(True)
    for emodel in tqdm(repaired_exemplar_df.emodel.unique()):
        exemplar_id = repaired_exemplar_df[
            (repaired_exemplar_df.emodel == emodel)
            & (repaired_exemplar_df.for_optimisation == 1)
        ].index

        if len(exemplar_id) > 1:
            raise Exception("Found more than one exemplar, something is wrong.")
        exemplar_id = exemplar_id[0]

        for mtype in repaired_df[(repaired_df.emodel == emodel)].mtype.unique():
            ids = repaired_df[
                (repaired_df.emodel == emodel) & (repaired_df.mtype == mtype)
            ].index
            for col in repaired_exemplar_score_df.columns:
                threshold = _get_threshold(
                    mtype,
                    emodel,
                    col,
                    megate_thresholds,
                    base_threshold=base_threshold,
                    no_threshold=no_threshold,
                )
                repaired_score_df.loc[ids, col] = repaired_score_df.loc[
                    ids, col
                ] <= max(
                    threshold,
                    threshold * repaired_exemplar_score_df.loc[exemplar_id, col],
                )
    return repaired_score_df


class ApplyGenericMegating(BaseTask):
    """Apply megating to scores."""

    emodels = luigi.ListParameter(default=["all"])
    megate_thresholds_path = luigi.Parameter(default="megate_thresholds.yaml")
    base_threshold = luigi.FloatParameter(default=5)
    no_threshold = luigi.FloatParameter(default=250)

    def requires(self):
        """Requires."""
        return {
            "exemplar": EvaluateExemplars(emodels=self.emodels),
            "generic": GatherGenericEvaluations(emodels=self.emodels),
        }

    def run(self):
        """Run."""
        selected_combos_df = pd.read_csv(self.input()["generic"].path)
        exemplar_combos_df = pd.read_csv(self.input()["exemplar"].path)
        megate_thresholds = yaml.full_load(open(self.megate_thresholds_path, "r"))

        megated_scores_df = apply_megating(
            selected_combos_df,
            exemplar_combos_df,
            megate_thresholds,
            self.base_threshold,
            self.no_threshold,
        )
        megated_scores_df.to_csv(self.output().path, index=False)


class ApplyMegating(BaseTask):
    """Apply megating to scores."""

    emodels = luigi.ListParameter(default=["all"])
    megate_thresholds_path = luigi.Parameter(default="megate_threshols.yaml")
    base_threshold = luigi.FloatParameter(default=5)
    no_threshold = luigi.FloatParameter(default=250)

    def requires(self):
        """Requires."""
        return {
            "exemplar": EvaluateExemplars(emodels=self.emodels),
            "synth": GatherSynthEvaluations(emodels=self.emodels),
        }

    def run(self):
        """Run."""
        selected_combos_df = pd.read_csv(self.input()["synth"].path)
        exemplar_combos_df = pd.read_csv(self.input()["exemplar"].path)
        megate_thresholds = yaml.full_load(open(self.megate_thresholds_path, "r"))
        megated_scores_df = apply_megating(
            selected_combos_df,
            exemplar_combos_df,
            megate_thresholds,
            self.base_threshold,
            self.no_threshold,
        )
        megated_scores_df.to_csv(self.output().path, index=False)


class SelectCombos(BaseTask):
    """Select combos with  scores under a threshold."""

    emodels = luigi.ListParameter(default=["all"])
    best_emodel_path = luigi.Parameter(default="best_emodels.csv")

    def requires(self):
        """Requires."""
        return {
            "megated": ApplyMegating(emodels=self.emodels),
            "synth": GatherSynthEvaluations(emodels=self.emodels),
        }

    def run(self):
        """Run."""
        selected_combos_df = pd.read_csv(self.input()["synth"].path)
        megated_scores_df = pd.read_csv(self.input()["megated"].path)

        selected_combos_df["selected"] = megated_scores_df.all(axis=1)

        _select_best_emodel(selected_combos_df, self.best_emodel_path)
        ensure_dir(self.output().path)
        selected_combos_df.to_csv(self.output().path, index=False)


class PlotSelected(BaseTask):
    """Plot report of cell selection."""

    emodels = luigi.ListParameter(default=["all"])

    def requires(self):
        """Requires."""
        return {
            "megated": ApplyMegating(emodels=self.emodels),
            "selected": SelectCombos(emodels=self.emodels),
        }

    def run(self):
        """Run."""
        ensure_dir(self.output().path)

        select_df = pd.read_csv(self.input()["selected"].path)
        megate_df = pd.read_csv(self.input()["megated"].path)
        with PdfPages(self.output().path) as pdf:

            plot_summary_select(select_df, e_column="etype")
            plt.suptitle("e-types with median scores")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            plot_summary_select(select_df, e_column="emodel")
            plt.suptitle("e-models with median scores")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            plot_feature_summary_select(select_df, megate_df, e_column="etype")
            plt.suptitle("failed features per e-types")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            plot_feature_summary_select(select_df, megate_df, e_column="emodel")
            plt.suptitle("failed features per e-models")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            plot_feature_select(select_df, megate_df, pdf, e_column="etype")
            plot_feature_select(select_df, megate_df, pdf, e_column="emodel")

            try:
                plot_frac_exceptions(select_df, e_column="mtype")
                pdf.savefig(bbox_inches="tight")
                plt.close()

                plot_frac_exceptions(select_df, e_column="emodel")
                pdf.savefig(bbox_inches="tight")
                plt.close()
            except IndexError:
                logger.info("No failed evaluation with exceptions!")


class SelectGenericCombos(BaseTask):
    """Select combos with  scores under a threshold."""

    emodels = luigi.ListParameter(default=["all"])
    best_emodel_path = luigi.Parameter(default="best_emodels.csv")

    def requires(self):
        """Requires."""
        return {
            "megated": ApplyGenericMegating(emodels=self.emodels),
            "generic": GatherGenericEvaluations(emodels=self.emodels),
        }

    def run(self):
        """Run."""
        selected_combos_df = pd.read_csv(self.input()["generic"].path)
        megated_scores_df = pd.read_csv(self.input()["megated"].path)

        selected_combos_df["selected"] = megated_scores_df.all(axis=1)

        _select_best_emodel(selected_combos_df, self.best_emodel_path)
        ensure_dir(self.output().path)
        selected_combos_df.to_csv(self.output().path, index=False)


class PlotGenericSelected(BaseTask):
    """Plot non selected cells."""

    emodels = luigi.ListParameter(default=["all"])

    def requires(self):
        """Requires."""
        return {
            "megated": ApplyGenericMegating(emodels=self.emodels),
            "selected": SelectGenericCombos(emodels=self.emodels),
        }

    def run(self):
        """Run."""
        ensure_dir(self.output().path)

        select_df = pd.read_csv(self.input()["selected"].path)
        megate_df = pd.read_csv(self.input()["megated"].path)
        with PdfPages(self.output().path) as pdf:

            plot_summary_select(select_df, e_column="etype")
            plt.suptitle("e-types with median scores")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            plot_summary_select(select_df, e_column="emodel")
            plt.suptitle("e-models with median scores")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            plot_feature_summary_select(select_df, megate_df, e_column="etype")
            plt.suptitle("failed features per e-types")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            plot_feature_summary_select(select_df, megate_df, e_column="emodel")
            plt.suptitle("failed features per e-models")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            plot_feature_select(select_df, megate_df, pdf, e_column="etype")
            plot_feature_select(select_df, megate_df, pdf, e_column="emodel")

            try:
                plot_frac_exceptions(select_df, e_column="mtype")
                pdf.savefig(bbox_inches="tight")
                plt.close()

                plot_frac_exceptions(select_df, e_column="emodel")
                pdf.savefig(bbox_inches="tight")
                plt.close()
            except IndexError:
                logger.info("No failed evaluation with exceptions!")
