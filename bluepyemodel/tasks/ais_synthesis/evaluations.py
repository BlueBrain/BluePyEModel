"""Tasks to evaluate scores on combos."""
import pandas as pd
import luigi

from .base_task import BaseTask
from ...ais_synthesis.evaluators import evaluate_combos_rho_scores
from .ais_synthesis import SynthesizeAis
from .utils import (
    add_emodel,
    ensure_dir,
)
from .config import morphologyconfigs


class EvaluateSynthesis(BaseTask):
    """Evaluate the cells witht synthesized AIS."""

    emodel = luigi.Parameter()
    save_traces = luigi.BoolParameter()
    eval_db_path = luigi.Parameter(default="eval_db.sql")
    morphology_path = luigi.Parameter(default="morphology_path")

    def requires(self):
        """Requires."""
        return {"synth_ais": SynthesizeAis(emodel=self.emodel)}

    def run(self):
        """Run."""

        synth_combos_df = pd.read_csv(self.input()["synth_ais"].path)

        synth_combos_with_scores_df = evaluate_combos_rho_scores(
            synth_combos_df,
            self.emodel_db,
            emodels=[self.emodel],
            morphology_path=self.morphology_path,
            continu=self.continu,
            ipyp_profile=self.ipyp_profile,
            save_traces=self.save_traces,
            combos_db_filename=add_emodel(self.eval_db_path, self.emodel),
        )

        ensure_dir(self.output().path)
        synth_combos_with_scores_df.to_csv(self.output().path, index=False)


class EvaluateGeneric(BaseTask):
    """Evaluate the cells witht synthesized AIS."""

    emodel = luigi.Parameter()
    save_traces = luigi.BoolParameter(default=False)
    eval_db_path = luigi.Parameter(default="eval_db.sql")
    morphology_path = luigi.Parameter(default="morphology_path")

    def run(self):
        """Run."""

        morphs_combos_df = pd.read_csv(morphologyconfigs().morphs_combos_df_path)
        morphs_combos_with_scores_df = evaluate_combos_rho_scores(
            morphs_combos_df,
            self.emodel_db,
            emodels=[self.emodel],
            morphology_path=self.morphology_path,
            continu=self.continu,
            ipyp_profile=self.ipyp_profile,
            save_traces=self.save_traces,
            combos_db_filename=add_emodel(self.eval_db_path, self.emodel),
        )

        ensure_dir(self.output().path)
        morphs_combos_with_scores_df.to_csv(self.output().path, index=False)
