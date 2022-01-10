"""Luigi tasks to gather all final ouptut from separate emodel runs."""
import luigi
import pandas as pd
import yaml

from bluepyemodel.tasks.generalisation.ais_model import AisResistanceModel
from bluepyemodel.tasks.generalisation.ais_model import AisShapeModel
from bluepyemodel.tasks.generalisation.ais_model import TargetRho
from bluepyemodel.tasks.generalisation.ais_model import TargetRhoAxon
from bluepyemodel.tasks.generalisation.ais_synthesis import SynthesizeAis
from bluepyemodel.tasks.generalisation.base_task import BaseTask
from bluepyemodel.tasks.generalisation.config import GatherLocalTarget
from bluepyemodel.tasks.generalisation.config import ScaleConfig
from bluepyemodel.tasks.generalisation.evaluations import EvaluateExemplars
from bluepyemodel.tasks.generalisation.evaluations import EvaluateGeneric
from bluepyemodel.tasks.generalisation.evaluations import EvaluateSynthesis
from bluepyemodel.tasks.generalisation.utils import ensure_dir


class GatherAisModels(BaseTask):
    """Gather all ais models in a single yaml file."""

    emodels = luigi.ListParameter()
    target_path = luigi.Parameter(default="ais_models.yaml")

    def requires(self):
        """Requires."""
        tasks = {"ais_shape_model": AisShapeModel()}
        for emodel in self.emodels:
            tasks["ais_res_model_" + emodel] = AisResistanceModel(emodel=emodel)
        return tasks

    def run(self):
        """Run."""
        with self.input()["ais_shape_model"].open() as f:
            ais_models = yaml.full_load(f)

        for emodel in self.emodels:
            with self.input()["ais_res_model_" + emodel].open() as f:
                ais_model = yaml.full_load(f)
                for mtype in ais_model:
                    if (
                        "resistance" in ais_model[mtype]
                        and emodel in ais_model[mtype]["resistance"]
                    ):
                        if "resistance" not in ais_models[mtype]:
                            ais_models[mtype]["resistance"] = {}
                        ais_models[mtype]["resistance"][emodel] = ais_model[mtype]["resistance"][
                            emodel
                        ]

        # record the scales_params for emodel_release
        ais_models_final = {
            "mtype": ais_models,
            "scales_params": ScaleConfig().scales_params,
        }

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(ais_models_final, f)

    def output(self):
        """ """
        return GatherLocalTarget(self.target_path)


class GatherTargetRho(BaseTask):
    """Gather all target rho in a single yaml file."""

    emodels = luigi.ListParameter()
    target_path = luigi.Parameter(default="target_rho.yaml")

    def requires(self):
        """Requires."""
        tasks = {}
        for emodel in self.emodels:
            tasks["target_rho_" + emodel] = TargetRho(emodel=emodel)
        return tasks

    def run(self):
        """Run."""
        target_rho = {}
        for emodel in self.emodels:
            with self.input()["target_rho_" + emodel].open() as f:
                rhos = yaml.full_load(f)
                target_rho[emodel] = rhos["rho"][emodel]

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(target_rho, f)

    def output(self):
        """ """
        return GatherLocalTarget(self.target_path)


class GatherTargetRhoAxon(BaseTask):
    """Gather all target rho axons in a single yaml file."""

    emodels = luigi.ListParameter()
    target_path = luigi.Parameter(default="target_rho_axons.yaml")

    def requires(self):
        """Requires."""
        tasks = {}
        for emodel in self.emodels:
            tasks["target_rho_axon_" + emodel] = TargetRhoAxon(emodel=emodel)
        return tasks

    def run(self):
        """Run."""
        target_rho_axons = {}
        for emodel in self.emodels:
            with self.input()["target_rho_axon_" + emodel].open() as f:
                rhos = yaml.full_load(f)
                target_rho_axons[emodel] = rhos["rho_axon"][emodel]

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(target_rho_axons, f)

    def output(self):
        """ """
        return GatherLocalTarget(self.target_path)


class GatherSynthAis(BaseTask):
    """Gather the synthesized AIS to final dataframe."""

    emodels = luigi.ListParameter()
    target_path = luigi.Parameter(default="synth_combos_df.csv")

    def requires(self):
        """ """
        return {emodel: SynthesizeAis(emodel=emodel) for emodel in self.emodels}

    def run(self):
        """ """
        synth_combos_df = pd.concat(
            [pd.read_csv(self.input()[emodel].path) for emodel in self.emodels]
        )

        ensure_dir(self.output().path)
        synth_combos_df.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return GatherLocalTarget(self.target_path)


class GatherExemplarEvaluations(BaseTask):
    """Gather all the evaluations to same final dataframe."""

    emodels = luigi.ListParameter()
    target_path = luigi.Parameter(default="exemplar_evaluations.csv")

    def requires(self):
        """ """
        return {emodel: EvaluateExemplars(emodel=emodel) for emodel in self.emodels}

    def run(self):
        """ """
        exemplar_combos_df = pd.concat(
            [pd.read_csv(self.input()[emodel].path) for emodel in self.emodels]
        )

        ensure_dir(self.output().path)
        exemplar_combos_df.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return GatherLocalTarget(self.target_path)


class GatherSynthEvaluations(BaseTask):
    """Gather all the evaluations to same final dataframe."""

    emodels = luigi.ListParameter()
    target_path = luigi.Parameter(default="synth_evaluations_combos_df.csv")

    def requires(self):
        """ """
        return {emodel: EvaluateSynthesis(emodel=emodel) for emodel in self.emodels}

    def run(self):
        """ """
        synth_eval_combos_df = pd.concat(
            [pd.read_csv(self.input()[emodel].path) for emodel in self.emodels]
        )

        ensure_dir(self.output().path)
        synth_eval_combos_df.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return GatherLocalTarget(self.target_path)


class GatherGenericEvaluations(BaseTask):
    """Gather all the evaluations to same final dataframe."""

    emodels = luigi.ListParameter()
    target_path = luigi.Parameter(default="evaluations_combos_df.csv")

    def requires(self):
        """ """
        return {emodel: EvaluateGeneric(emodel=emodel) for emodel in self.emodels}

    def run(self):
        """Run."""
        morphs_combos_df = pd.concat(
            [pd.read_csv(self.input()[emodel].path) for emodel in self.emodels]
        )

        ensure_dir(self.output().path)
        morphs_combos_df.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return GatherLocalTarget(self.target_path)
