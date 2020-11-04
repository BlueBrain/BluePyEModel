"""Luigi tasks to gather all final ouptut from separate emodel runs."""
import pandas as pd
import yaml

import luigi

from .ais_model import AisResistanceModel, AisShapeModel, TargetRhoAxon
from .ais_synthesis import SynthesizeAis
from .base_task import BaseTask
from .config import scaleconfigs
from .evaluations import EvaluateGeneric, EvaluateSynthesis
from .utils import ensure_dir


class GatherAisModels(BaseTask):
    """Gather all ais models in a single yaml file."""

    emodels = luigi.ListParameter()

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
            "scales_params": scaleconfigs().scales_params,
        }

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(ais_models_final, f)


class GatherTargetRhoAxon(BaseTask):
    """Gather all target rho axons in a single yaml file."""

    emodels = luigi.ListParameter()

    def requires(self):
        """Requires."""
        tasks = {}
        for emodel in self.emodels:
            tasks["target_rho_" + emodel] = TargetRhoAxon(emodel=emodel)
        return tasks

    def run(self):
        """Run."""
        target_rhos = {}
        for emodel in self.emodels:
            with self.input()["target_rho_" + emodel].open() as f:
                target_rhos[emodel] = yaml.full_load(f)[emodel]

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(target_rhos, f)


class GatherSynthAis(BaseTask):
    """Gather the synthesized AIS to final dataframe."""

    emodels = luigi.ListParameter()

    def requires(self):
        """"""
        return {emodel: SynthesizeAis(emodel=emodel) for emodel in self.emodels}

    def run(self):
        """"""
        synth_combos_df = pd.concat(
            [pd.read_csv(self.input()[emodel].path) for emodel in self.emodels]
        )

        ensure_dir(self.output().path)
        synth_combos_df.to_csv(self.output().path, index=False)


class GatherSynthEvaluations(BaseTask):
    """Gather all the evaluations to same final dataframe."""

    emodels = luigi.ListParameter()

    def requires(self):
        """"""
        return {emodel: EvaluateSynthesis(emodel=emodel) for emodel in self.emodels}

    def run(self):
        """"""
        synth_eval_combos_df = pd.concat(
            [pd.read_csv(self.input()[emodel].path) for emodel in self.emodels]
        )

        ensure_dir(self.output().path)
        synth_eval_combos_df.to_csv(self.output().path, index=False)


class GatherGenericEvaluations(BaseTask):
    """Gather all the evaluations to same final dataframe."""

    emodels = luigi.ListParameter()

    def requires(self):
        """"""
        return {emodel: EvaluateGeneric(emodel=emodel) for emodel in self.emodels}

    def run(self):
        """Run."""
        morphs_combos_df = pd.concat(
            [pd.read_csv(self.input()[emodel].path) for emodel in self.emodels]
        )

        ensure_dir(self.output().path)
        morphs_combos_df.to_csv(self.output().path, index=False)
