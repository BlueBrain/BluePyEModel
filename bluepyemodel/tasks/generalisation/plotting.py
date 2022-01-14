"""Plotting Luigi tasks to run the ais workflow."""
import logging
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from bluepyemodel.generalisation.plotting import plot_ais_resistance_models
from bluepyemodel.generalisation.plotting import plot_ais_taper_models
from bluepyemodel.generalisation.plotting import plot_feature_select
from bluepyemodel.generalisation.plotting import plot_feature_summary_select
from bluepyemodel.generalisation.plotting import plot_frac_exceptions
from bluepyemodel.generalisation.plotting import plot_soma_resistance_models
from bluepyemodel.generalisation.plotting import plot_soma_shape_models
from bluepyemodel.generalisation.plotting import plot_summary_select
from bluepyemodel.generalisation.plotting import plot_surface_comparison
from bluepyemodel.generalisation.plotting import plot_synth_ais_evaluations
from bluepyemodel.generalisation.plotting import plot_target_rhos
from bluepyemodel.tasks.generalisation.ais_model import AisResistanceModel
from bluepyemodel.tasks.generalisation.ais_model import AisShapeModel
from bluepyemodel.tasks.generalisation.ais_model import SomaResistanceModel
from bluepyemodel.tasks.generalisation.ais_model import SomaShapeModel
from bluepyemodel.tasks.generalisation.ais_model import TargetRho
from bluepyemodel.tasks.generalisation.ais_model import TargetRhoAxon
from bluepyemodel.tasks.generalisation.base_task import BaseTask
from bluepyemodel.tasks.generalisation.config import PlotLocalTarget
from bluepyemodel.tasks.generalisation.evaluations import EvaluateGeneric
from bluepyemodel.tasks.generalisation.evaluations import EvaluateSynthesis
from bluepyemodel.tasks.generalisation.select import ApplyGenericMegating
from bluepyemodel.tasks.generalisation.select import ApplyMegating
from bluepyemodel.tasks.generalisation.select import SelectCombos
from bluepyemodel.tasks.generalisation.select import SelectGenericCombos
from bluepyemodel.tasks.generalisation.utils import ensure_dir

logger = logging.getLogger(__name__)


class PlotSomaShapeModel(BaseTask):
    """Plot the soma shape models."""

    target_path = luigi.Parameter(default="soma_shape_models.pdf")

    def requires(self):
        """ """
        return SomaShapeModel()

    def run(self):
        """ """
        ensure_dir(self.output().path)
        with self.input().open() as soma_model_file:
            plot_soma_shape_models(yaml.safe_load(soma_model_file), self.output().path)

    def output(self):
        """ """
        return PlotLocalTarget(self.target_path)


class PlotSomaResistanceModel(BaseTask):
    """Plot the soma shape models."""

    emodel = luigi.Parameter()
    target_path = luigi.Parameter(default="resistance_models/soma_resistance_model.pdf")

    def requires(self):
        """ """
        return SomaResistanceModel(emodel=self.emodel)

    def run(self):
        """ """
        _task = SomaResistanceModel(emodel=self.emodel)
        fit_df = pd.read_csv(_task.set_tmp(self.add_emodel(_task.fit_df_path)))
        ensure_dir(self.output().path)
        with self.input().open() as soma_models_file:
            plot_soma_resistance_models(
                fit_df,
                yaml.safe_load(soma_models_file),
                pdf_filename=self.output().path,
            )

    def output(self):
        """ """
        return PlotLocalTarget(self.add_emodel(self.target_path))


class PlotAisShapeModel(BaseTask):
    """Plot the AIS shape models."""

    target_path = luigi.Parameter(default="AIS_shape_models.pdf")

    def requires(self):
        """ """
        return AisShapeModel()

    def run(self):
        """ """
        ensure_dir(self.output().path)
        with self.input().open() as ais_model_file:
            plot_ais_taper_models(yaml.safe_load(ais_model_file), self.output().path)

    def output(self):
        """ """
        return PlotLocalTarget(self.target_path)


class PlotAisResistanceModel(BaseTask):
    """Plot the AIS shape models."""

    emodel = luigi.Parameter()
    target_path = luigi.Parameter(default="resistance_models/AIS_resistance_model.pdf")

    def requires(self):
        """ """
        return AisResistanceModel(emodel=self.emodel)

    def run(self):
        """ """
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
        """ """
        return PlotLocalTarget(self.add_emodel(self.target_path))


class PlotTargetRhoAxon(BaseTask):
    """Plot the scan of scales and target rhow axon."""

    emodel = luigi.Parameter()
    target_path = luigi.Parameter(default="target_rhos/target_rhos_axon.pdf")

    def requires(self):
        """ """
        return {
            "target_rho_axon": TargetRhoAxon(emodel=self.emodel),
            "target_rho": TargetRho(emodel=self.emodel),
            "evaluation": EvaluateGeneric(emodel=self.emodel),
        }

    def run(self):
        """ """
        _task = TargetRhoAxon(emodel=self.emodel)
        rho_axon_df = pd.read_csv(_task.set_tmp(self.add_emodel(_task.rho_axon_df_path)))

        _task = TargetRho(emodel=self.emodel)
        rho_df = pd.read_csv(_task.set_tmp(self.add_emodel(_task.rho_df_path)))

        df = pd.read_csv(self.input()["evaluation"].path)

        ensure_dir(self.output().path)
        with self.input()["target_rho"].open() as target_rhos_file:
            target_rhos = yaml.safe_load(target_rhos_file)
        with self.input()["target_rho_axon"].open() as target_rho_axons_file:
            target_rho_axons = yaml.safe_load(target_rho_axons_file)

        df["rho"] = rho_df["rho"]
        df["rho_axon"] = rho_axon_df["rho_axon"]
        plot_target_rhos(
            df,
            target_rhos,
            target_rho_axons,
            original_morphs_combos_path=None,
            pdf_filename=self.output().path,
        )

    def output(self):
        """ """
        return PlotLocalTarget(self.add_emodel(self.target_path))


class PlotSynthesisEvaluation(BaseTask):
    """Plot the evaluation results of synthesized cells."""

    emodel = luigi.Parameter()
    threshold = luigi.FloatParameter(default=5)
    target_path = luigi.Parameter(default="evaluations/synthesis_evaluation.pdf")

    def requires(self):
        """ """
        return EvaluateSynthesis(emodel=self.emodel)

    def run(self):
        """ """
        ensure_dir(self.output().path)
        plot_synth_ais_evaluations(
            pd.read_csv(self.input().path),
            emodels=[self.emodel],
            threshold=self.threshold,
            pdf_filename=self.output().path,
        )

    def output(self):
        """ """
        return PlotLocalTarget(self.add_emodel(self.target_path))


class PlotSurfaceComparison(BaseTask):
    """Make the surface area comparison plots."""

    emodel = luigi.Parameter()
    threshold = luigi.FloatParameter(default=5)
    target_path = luigi.Parameter(default="surfaces_comparison/surface.pdf")

    def requires(self):
        """ """
        return EvaluateSynthesis(emodel=self.emodel)

    def run(self):
        """ """
        ensure_dir(self.output().path)
        df = pd.read_csv(self.input().path)
        plot_surface_comparison(
            df[df.emodel == self.emodel],
            pdf_filename=self.output().path,
            bin_params={"min": 0, "max": 1500, "n": 100},
        )

    def output(self):
        """ """
        return PlotLocalTarget(self.add_emodel(self.target_path))


class PlotGenericEvaluation(BaseTask):
    """Plot the evaluation results of synthesized cells."""

    emodel = luigi.Parameter()
    threshold = luigi.FloatParameter(default=5)
    target_path = luigi.Parameter(default="evaluations/evaluation.pdf")

    def requires(self):
        """ """
        return EvaluateGeneric(emodel=self.emodel)

    def run(self):
        """ """
        ensure_dir(self.output().path)
        plot_synth_ais_evaluations(
            pd.read_csv(self.input().path),
            emodels=[self.emodel],
            threshold=self.threshold,
            pdf_filename=self.output().path,
        )

    def output(self):
        """ """
        return PlotLocalTarget(self.add_emodel(self.target_path))


class PlotSelected(BaseTask):
    """Plot report of cell selection."""

    emodels = luigi.ListParameter()
    target_path = luigi.Parameter(default="select_summary.pdf")

    def requires(self):
        """ """
        return {
            "megated": ApplyMegating(emodels=self.emodels),
            "selected": SelectCombos(emodels=self.emodels),
        }

    def run(self):
        """ """

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
        """ """
        return PlotLocalTarget(self.target_path)


class PlotGenericSelected(BaseTask):
    """Plot non selected cells."""

    emodels = luigi.ListParameter()
    target_path = luigi.Parameter(default="select_summary.pdf")

    def requires(self):
        """ """
        return {
            "megated": ApplyGenericMegating(emodels=self.emodels),
            "selected": SelectGenericCombos(emodels=self.emodels),
        }

    def run(self):
        """ """

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
        """ """
        return PlotLocalTarget(self.target_path)
