"""Plotting Luigi tasks to run the ais workflow."""
import logging
from pathlib import Path
import yaml
import pandas as pd

import luigi
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ...ais_synthesis.plotting import (
    plot_ais_resistance_models,
    plot_ais_taper_models,
    plot_synth_ais_evaluations,
    plot_target_rho_axon,
)
from .ais_model import AisResistanceModel, AisShapeModel, TargetRhoAxon
from .base_task import BaseTask
from .evaluations import EvaluateGeneric, EvaluateSynthesis
from .select import ApplyMegating, ApplyGenericMegating, SelectCombos, SelectGenericCombos
from ...ais_synthesis.plotting import (
    plot_feature_select,
    plot_feature_summary_select,
    plot_frac_exceptions,
    plot_summary_select,
)
from .utils import ensure_dir
from .config import PlotLocalTarget

logger = logging.getLogger(__name__)


class PlotAisShapeModel(BaseTask):
    """Plot the AIS shape models."""

    target_path = luigi.Parameter(default="AIS_shape_models.pdf")

    def requires(self):
        """"""
        return AisShapeModel()

    def run(self):
        """"""
        ensure_dir(self.output().path)
        with self.input().open() as ais_model_file:
            plot_ais_taper_models(yaml.safe_load(ais_model_file), self.output().path)

    def output(self):
        """"""
        return PlotLocalTarget(self.target_path)


class PlotAisResistanceModel(BaseTask):
    """Plot the AIS shape models."""

    emodel = luigi.Parameter()
    target_path = luigi.Parameter(default="resistance_models/AIS_resistance_model.pdf")

    def requires(self):
        """"""
        return AisResistanceModel(emodel=self.emodel)

    def run(self):
        """"""
        _task = AisResistanceModel(emodel=self.emodel)
        fit_df = pd.read_csv(_task.set_tmp(self.add_emodel(_task.fit_df_path)))
        ensure_dir(self.output().path)
        with self.input().open() as ais_models_file:
            plot_ais_resistance_models(
                fit_df,
                yaml.safe_load(ais_models_file),
                pdf_filename=self.output().path,
            )

    def output(self):
        """"""
        return PlotLocalTarget(self.add_emodel(self.target_path))


class PlotTargetRhoAxon(BaseTask):
    """Plot the scan of scales and target rhow axon."""

    emodel = luigi.Parameter()
    target_path = luigi.Parameter(default="target_rhos/target_rhos_axon.pdf")

    def requires(self):
        """"""
        return TargetRhoAxon(emodel=self.emodel)

    def run(self):
        """"""
        try:
            _task = TargetRhoAxon(emodel=self.emodel)
            rho_scan_df = pd.read_csv(_task.set_tmp(self.add_emodel(_task.rho_scan_df_path)))

            ensure_dir(self.output().path)
            with self.input().open() as target_rhos_file:
                plot_target_rho_axon(
                    rho_scan_df,
                    yaml.safe_load(target_rhos_file),
                    original_morphs_combos_path=None,
                    pdf_filename=self.output().path,
                )
        except FileNotFoundError:
            ensure_dir(self.output().path)
            f = open(self.output().path, "w")
            f.write("no plot for linear_fit mode")

    def output(self):
        """"""
        return PlotLocalTarget(self.add_emodel(self.target_path))


class PlotSynthesisEvaluation(BaseTask):
    """Plot the evaluation results of synthesized cells."""

    emodel = luigi.Parameter()
    threshold = luigi.FloatParameter(default=5)
    target_path = luigi.Parameter(default="evaluations/synthesis_evaluation.pdf")

    def requires(self):
        """"""
        return EvaluateSynthesis(emodel=self.emodel)

    def run(self):
        """"""
        ensure_dir(self.output().path)
        plot_synth_ais_evaluations(
            pd.read_csv(self.input().path),
            emodels=[self.emodel],
            threshold=self.threshold,
            pdf_filename=self.output().path,
        )

    def output(self):
        """"""
        return PlotLocalTarget(self.add_emodel(self.target_path))


class PlotGenericEvaluation(BaseTask):
    """Plot the evaluation results of synthesized cells."""

    emodel = luigi.Parameter()
    threshold = luigi.FloatParameter(default=5)
    target_path = luigi.Parameter(default="evaluations/evaluation.pdf")

    def requires(self):
        """"""
        return EvaluateGeneric(emodel=self.emodel)

    def run(self):
        """"""
        ensure_dir(self.output().path)
        plot_synth_ais_evaluations(
            pd.read_csv(self.input().path),
            emodels=[self.emodel],
            threshold=self.threshold,
            pdf_filename=self.output().path,
        )

    def output(self):
        """"""
        return PlotLocalTarget(self.add_emodel(self.target_path))


class PlotSelected(BaseTask):
    """Plot report of cell selection."""

    emodels = luigi.ListParameter()
    target_path = luigi.Parameter(default="select_summary.pdf")

    def requires(self):
        """"""
        return {
            "megated": ApplyMegating(emodels=self.emodels),
            "selected": SelectCombos(emodels=self.emodels),
        }

    def run(self):
        """"""

        select_df = pd.read_csv(self.input()["selected"].path)
        megate_df = pd.read_csv(self.input()["megated"].path)
        ensure_dir(self.output().path)
        with PdfPages(self.output().path) as pdf:

            plot_summary_select(select_df, e_column="etype")
            plt.suptitle("e-types with median scores")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            plot_summary_select(select_df, e_column="emodel")
            plt.suptitle("e-models with median scores")
            pdf.savefig(bbox_inches="tight")
            plt.close()

        with PdfPages(str(Path(self.output().path).with_suffix("")) + "_features.pdf") as pdf:
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

    def output(self):
        """"""
        return PlotLocalTarget(self.target_path)


class PlotGenericSelected(BaseTask):
    """Plot non selected cells."""

    emodels = luigi.ListParameter()

    def requires(self):
        """"""
        return {
            "megated": ApplyGenericMegating(emodels=self.emodels),
            "selected": SelectGenericCombos(emodels=self.emodels),
        }

    def run(self):
        """"""

        select_df = pd.read_csv(self.input()["selected"].path)
        megate_df = pd.read_csv(self.input()["megated"].path)

        ensure_dir(self.output().path)
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

    def output(self):
        """"""
        return PlotLocalTarget(self.target_path)
