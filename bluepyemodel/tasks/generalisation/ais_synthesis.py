"""Tasks to synthesize ais"""
import luigi
import pandas as pd
import yaml
from bluepyparallel import init_parallel_factory

from bluepyemodel.generalisation.ais_synthesis import synthesize_ais
from bluepyemodel.generalisation.ais_synthesis import synthesize_soma
from bluepyemodel.tasks.generalisation.ais_model import AisResistanceModel
from bluepyemodel.tasks.generalisation.ais_model import SomaResistanceModel
from bluepyemodel.tasks.generalisation.ais_model import TargetRho
from bluepyemodel.tasks.generalisation.ais_model import TargetRhoAxon
from bluepyemodel.tasks.generalisation.base_task import BaseTask
from bluepyemodel.tasks.generalisation.config import ScaleConfig
from bluepyemodel.tasks.generalisation.config import SynthesisLocalTarget
from bluepyemodel.tasks.generalisation.morph_combos import CreateMorphCombosDF
from bluepyemodel.tasks.generalisation.utils import ensure_dir


class SynthesizeAisSoma(BaseTask):
    """Synthesize AIS and Soma."""
    emodel = luigi.Parameter(default=None)
    morphology_path = luigi.Parameter(default="morphology_path")
    synth_db_path = luigi.Parameter(default="synth_db.sql")
    target_path = luigi.Parameter(default="synth_combos_df.csv")
    n_steps = luigi.IntParameter(default=3)
    with_soma = luigi.BoolParameter(default=True)

    def requires(self):
        """Requires."""
        tasks = {
            "ais_models": AisResistanceModel(emodel=self.emodel),
            "target_rho_axons": TargetRhoAxon(emodel=self.emodel),
            "morph_combos": CreateMorphCombosDF(),
        }
        if self.with_soma:
            tasks.update(
                {
                    "soma_models": SomaResistanceModel(emodel=self.emodel),
                    "target_rhos": TargetRho(emodel=self.emodel),
                }
            )
        return tasks

    def run(self):
        """Run."""
        morphs_combos_df = pd.read_csv(self.input()["morph_combos"].path)
        if self.emodel is not None:
            morphs_combos_df = morphs_combos_df[morphs_combos_df.emodel == self.emodel]

        with self.input()["target_rho_axons"].open() as f:
            target_rho_axons = yaml.full_load(f)

        with self.input()["ais_models"].open() as f:
            ais_models = yaml.full_load(f)

        if self.with_soma:
            self.n_steps = 1
            with self.input()["target_rhos"].open() as f:
                target_rhos = yaml.full_load(f)

            with self.input()["soma_models"].open() as f:
                soma_models = yaml.full_load(f)

        ensure_dir(self.set_tmp(self.add_emodel(self.synth_db_path)))
        parallel_factory = init_parallel_factory(self.parallel_lib)
        for _ in range(self.n_steps):
            if "rin_no_soma" in morphs_combos_df.columns:
                morphs_combos_df = morphs_combos_df.drop(columns=["rin_no_soma"])
            if "rin_no_axon" in morphs_combos_df.columns:
                morphs_combos_df = morphs_combos_df.drop(columns=["rin_no_axon"])

            if self.with_soma:
                morphs_combos_df = synthesize_soma(
                    morphs_combos_df,
                    self.emodel_db,
                    soma_models,
                    target_rhos,
                    morphology_path=self.morphology_path,
                    emodels=[self.emodel],
                    resume=self.resume,
                    parallel_factory=parallel_factory,
                    scales_params=ScaleConfig().scales_params,
                    db_url=self.set_tmp(self.add_emodel(self.synth_db_path)),
                )
            morphs_combos_df = synthesize_ais(
                morphs_combos_df,
                self.emodel_db,
                ais_models,
                target_rho_axons,
                morphology_path=self.morphology_path,
                emodels=[self.emodel],
                resume=self.resume,
                parallel_factory=parallel_factory,
                scales_params=ScaleConfig().scales_params,
                db_url=self.set_tmp(self.add_emodel(self.synth_db_path)),
            )
        ensure_dir(self.output().path)
        morphs_combos_df.to_csv(self.output().path, index=False)

        parallel_factory.shutdown()

    def output(self):
        """ """
        return SynthesisLocalTarget(self.add_emodel(self.target_path))
