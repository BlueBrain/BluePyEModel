"""Luigi tasks to gather all final ouptut from separate emodel runs."""
import pandas as pd
import yaml

import luigi

from .ais_model import (
    AisResistanceModel,
    AisShapeModel,
    TargetRhoAxon,
)
from .evaluations import (
    EvaluateGeneric,
    EvaluateSynthesis,
)
from .ais_synthesis import SynthesizeAis
from .base_task import BaseTask
from .utils import ensure_dir
from .config import morphologyconfigs, scaleconfigs


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
                        ais_models[mtype]["resistance"][emodel] = ais_model[mtype][
                            "resistance"
                        ][emodel]

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
        """Requires."""
        tasks = {}
        for emodel in self.emodels:
            tasks["synth_ais_" + emodel] = SynthesizeAis(emodel=emodel)
        return tasks

    def run(self):
        """Run."""
        synth_combos_df = pd.read_csv(morphologyconfigs().morphs_combos_df_path)
        synth_combos_df["AIS"] = ""
        for emodel in self.emodels:
            synth_combos_df.update(
                pd.read_csv(self.input()["synth_ais_" + emodel].path)
            )

        ensure_dir(self.output().path)
        synth_combos_df.to_csv(self.output().path, index=False)


class GatherSynthEvaluations(BaseTask):
    """Gather all the evaluations to same final dataframe."""

    emodels = luigi.ListParameter()

    def requires(self):
        """Requires."""
        tasks = {"synth_ais": GatherSynthAis(emodels=self.emodels)}
        for emodel in self.emodels:
            tasks["synth_eval_" + emodel] = EvaluateSynthesis(emodel=emodel)
        return tasks

    def run(self):
        """Run."""
        synth_eval_combos_df = pd.read_csv(self.input()["synth_ais"].path)
        for emodel in self.emodels:
            synth_eval_combo_df = pd.read_csv(self.input()["synth_eval_" + emodel].path)
            for col in synth_eval_combo_df.columns:
                if col not in synth_eval_combos_df.columns:
                    synth_eval_combos_df[col] = ""
            synth_eval_combos_df.update(synth_eval_combo_df)

        ensure_dir(self.output().path)
        synth_eval_combos_df.to_csv(self.output().path, index=False)


class GatherGenericEvaluations(BaseTask):
    """Gather all the evaluations to same final dataframe."""

    emodels = luigi.ListParameter()

    def requires(self):
        """Requires."""
        tasks = {}
        for emodel in self.emodels:
            tasks["generic_eval_" + emodel] = EvaluateGeneric(emodel=emodel)
        return tasks

    def run(self):
        """Run."""
        morphs_combos_df = pd.read_csv(morphologyconfigs().morphs_combos_df_path)
        for emodel in self.emodels:
            morphs_combo_df = pd.read_csv(self.input()["generic_eval_" + emodel].path)
            for col in morphs_combo_df.columns:
                if col not in morphs_combos_df.columns:
                    morphs_combos_df[col] = ""
            morphs_combos_df.update(morphs_combo_df)

        ensure_dir(self.output().path)
        morphs_combos_df.to_csv(self.output().path, index=False)
