"""Tasks to synthesize ais"""
import pandas as pd
import yaml

import luigi
from bluepyemodel.ais_synthesis.tools.parallel import InitParallelFactory

from ...ais_synthesis.ais_synthesis import synthesize_ais
from .ais_model import AisResistanceModel, TargetRhoAxon
from .base_task import BaseTask
from .config import scaleconfigs
from .morph_combos import CreateMorphCombosDF
from .utils import add_emodel, ensure_dir


class SynthesizeAis(BaseTask):
    """Synthesize AIS."""

    emodel = luigi.Parameter(default=None)
    morphology_path = luigi.Parameter(default="morphology_path")
    synth_db_path = luigi.Parameter(default="synth_db.sql")

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

        with InitParallelFactory(self.parallel_lib) as parallel_factory:
            synth_combos_df = synthesize_ais(
                morphs_combos_df,
                self.emodel_db,
                ais_models,
                target_rhos,
                morphology_path=self.morphology_path,
                emodels=[self.emodel],
                continu=self.continu,
                parallel_factory=parallel_factory,
                scales_params=scaleconfigs().scales_params,
                combos_db_filename=add_emodel(self.synth_db_path, self.emodel),
            )

        ensure_dir(self.output().path)
        synth_combos_df.to_csv(self.output().path, index=False)
