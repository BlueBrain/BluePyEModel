"""Luigi task for select step with plotting tasks."""
import logging

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

import luigi

from ...ais_synthesis.plotting import (
    plot_feature_select,
    plot_feature_summary_select,
    plot_frac_exceptions,
    plot_summary_select,
)
from ...ais_synthesis.select import apply_megating, select_best_emodel
from .base_task import BaseTask
from .config import selectconfigs
from .evaluations import EvaluateExemplars
from .gather import GatherGenericEvaluations, GatherSynthEvaluations
from .utils import ensure_dir

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class ApplyGenericMegating(BaseTask):
    """Apply megating to scores."""

    emodels = luigi.ListParameter()
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
        megate_thresholds = yaml.full_load(open(selectconfigs().megate_thresholds_path, "r"))

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

    emodels = luigi.ListParameter()
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
        megate_thresholds = yaml.full_load(open(selectconfigs().megate_thresholds_path, "r"))
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

    emodels = luigi.ListParameter()
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

        select_best_emodel(selected_combos_df, self.best_emodel_path)
        ensure_dir(self.output().path)
        selected_combos_df.to_csv(self.output().path, index=False)


class PlotSelected(BaseTask):
    """Plot report of cell selection."""

    emodels = luigi.ListParameter()

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

    emodels = luigi.ListParameter()
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

        select_best_emodel(selected_combos_df, self.best_emodel_path)
        ensure_dir(self.output().path)
        selected_combos_df.to_csv(self.output().path, index=False)


class PlotGenericSelected(BaseTask):
    """Plot non selected cells."""

    emodels = luigi.ListParameter()

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
