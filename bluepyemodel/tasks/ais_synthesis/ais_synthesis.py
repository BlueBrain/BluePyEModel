"""Tasks to synthesize ais"""
import luigi
import pandas as pd
import yaml

from bluepyemodel.ais_synthesis.ais_synthesis import synthesize_ais
from bluepyemodel.ais_synthesis.tools import init_parallel_factory
from bluepyemodel.tasks.ais_synthesis.ais_model import AisResistanceModel
from bluepyemodel.tasks.ais_synthesis.ais_model import TargetRhoAxon
from bluepyemodel.tasks.ais_synthesis.base_task import BaseTask
from bluepyemodel.tasks.ais_synthesis.config import ScaleConfig
from bluepyemodel.tasks.ais_synthesis.config import SynthesisLocalTarget
from bluepyemodel.tasks.ais_synthesis.morph_combos import CreateMorphCombosDF
from bluepyemodel.tasks.ais_synthesis.utils import ensure_dir


class SynthesizeAis(BaseTask):
    """Synthesize AIS."""

    emodel = luigi.Parameter(default=None)
    morphology_path = luigi.Parameter(default="repaired_morphology_path")
    synth_db_path = luigi.Parameter(default="synth_db.sql")
    target_path = luigi.Parameter(default="synth_combos_df.csv")

    def requires(self):
        """Requires."""
        tasks = {
            "ais_models": AisResistanceModel(emodel=self.emodel),
            "target_rhos": TargetRhoAxon(emodel=self.emodel),
            "morph_combos": CreateMorphCombosDF(),
        }
        return tasks

    def run(self):
        """Run."""
        morphs_combos_df = pd.read_csv(self.input()["morph_combos"].path)
        if self.emodel is not None:
            morphs_combos_df = morphs_combos_df[morphs_combos_df.emodel == self.emodel]

        with self.input()["target_rhos"].open() as f:
            target_rhos = yaml.full_load(f)

        with self.input()["ais_models"].open() as f:
            ais_models = yaml.full_load(f)

        ensure_dir(self.set_tmp(self.add_emodel(self.synth_db_path)))
        parallel_factory = init_parallel_factory(self.parallel_lib)
        synth_combos_df = synthesize_ais(
            morphs_combos_df,
            self.emodel_db,
            ais_models,
            target_rhos,
            morphology_path=self.morphology_path,
            emodels=[self.emodel],
            continu=self.continu,
            parallel_factory=parallel_factory,
            scales_params=ScaleConfig().scales_params,
            combos_db_filename=self.set_tmp(self.add_emodel(self.synth_db_path)),
        )

        ensure_dir(self.output().path)
        synth_combos_df.to_csv(self.output().path, index=False)

        parallel_factory.shutdown()

    def output(self):
        """"""
        return SynthesisLocalTarget(self.add_emodel(self.target_path))
