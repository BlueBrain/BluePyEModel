"""Plotting Luigi tasks to run the ais workflow."""
import pandas as pd
import yaml

import luigi

from ...ais_synthesis.plotting import (
    plot_ais_resistance_models,
    plot_ais_taper_models,
    plot_synth_ais_evaluations,
    plot_target_rho_axon,
)
from .ais_model import AisResistanceModel, AisShapeModel, TargetRhoAxon
from .base_task import BaseTask
from .evaluations import EvaluateGeneric, EvaluateSynthesis
from .utils import add_emodel, ensure_dir


class PlotAisShapeModel(BaseTask):
    """Plot the AIS shape models."""

    def requires(self):
        """Requires."""
        return AisShapeModel()

    def run(self):
        """Run."""
        ensure_dir(self.output().path)
        with self.input().open() as ais_model_file:
            plot_ais_taper_models(yaml.safe_load(ais_model_file), self.output().path)


class PlotAisResistanceModel(BaseTask):
    """Plot the AIS shape models."""

    emodel = luigi.Parameter()

    def requires(self):
        """Requires."""
        return AisResistanceModel(emodel=self.emodel)

    def run(self):
        """Run."""
        fit_df = pd.read_csv(
            add_emodel(AisResistanceModel(emodel=self.emodel).fit_df_path, self.emodel)
        )
        ensure_dir(self.output().path)
        with self.input().open() as ais_models_file:
            plot_ais_resistance_models(
                fit_df,
                yaml.safe_load(ais_models_file),
                pdf_filename=self.output().path,
            )


class PlotTargetRhoAxon(BaseTask):
    """Plot the scan of scales and target rhow axon."""

    emodel = luigi.Parameter()

    def requires(self):
        """Requires."""
        return TargetRhoAxon(emodel=self.emodel)

    def run(self):
        """Run."""
        try:
            rho_scan_df = pd.read_csv(
                add_emodel(TargetRhoAxon(emodel=self.emodel).rho_scan_df_path, self.emodel)
            )

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


class PlotSynthesisEvaluation(BaseTask):
    """Plot the evaluation results of synthesized cells."""

    emodel = luigi.Parameter()
    threshold = luigi.FloatParameter(default=5)

    def requires(self):
        """Requires."""
        return EvaluateSynthesis(emodel=self.emodel)

    def run(self):
        """Run."""
        ensure_dir(self.output().path)
        plot_synth_ais_evaluations(
            pd.read_csv(self.input().path),
            emodels=[self.emodel],
            threshold=self.threshold,
            pdf_filename=self.output().path,
        )


class PlotGenericEvaluation(BaseTask):
    """Plot the evaluation results of synthesized cells."""

    emodel = luigi.Parameter()
    threshold = luigi.FloatParameter(default=5)

    def requires(self):
        """Requires."""
        return EvaluateGeneric(emodel=self.emodel)

    def run(self):
        """Run."""
        ensure_dir(self.output().path)
        plot_synth_ais_evaluations(
            pd.read_csv(self.input().path),
            emodels=[self.emodel],
            threshold=self.threshold,
            pdf_filename=self.output().path,
        )
