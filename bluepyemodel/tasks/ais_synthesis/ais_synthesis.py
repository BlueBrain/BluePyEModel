"""Tasks to synthesize ais"""
import pandas as pd
import yaml

import luigi

from ...ais_synthesis.ais_synthesis import synthesize_ais
from .ais_model import AisResistanceModel, TargetRhoAxon
from .base_task import BaseTask
from .utils import (
    add_emodel,
    ensure_dir,
)
from .config import morphologyconfigs, scaleconfigs


class SynthesizeAis(BaseTask):
    """Synthesize AIS."""

    emodel = luigi.Parameter()
    morphology_path = luigi.Parameter(default="morphology_path")
    synth_db_path = luigi.Parameter(default="synth_db.sql")

    def requires(self):
        """Requires."""
        tasks = {
            "ais_models": AisResistanceModel(emodel=self.emodel),
            "target_rhos": TargetRhoAxon(emodel=self.emodel),
        }
        return self.add_dask_task(tasks)

    def run(self):
        """Run."""

        morphs_combos_df = pd.read_csv(morphologyconfigs().morphs_combos_df_path)

        with self.input()["target_rhos"].open() as f:
            target_rhos = yaml.full_load(f)

        with self.input()["ais_models"].open() as f:
            ais_models = yaml.full_load(f)

        synth_combos_df = synthesize_ais(
            morphs_combos_df,
            self.emodel_db,
            ais_models,
            target_rhos,
            morphology_path=self.morphology_path,
            emodels=[self.emodel],
            continu=self.continu,
            ipyp_profile=self.ipyp_profile,
            scales_params=scaleconfigs().scales_params,
            combos_db_filename=add_emodel(self.synth_db_path, self.emodel),
        )

        ensure_dir(self.output().path)
        synth_combos_df.to_csv(self.output().path, index=False)
