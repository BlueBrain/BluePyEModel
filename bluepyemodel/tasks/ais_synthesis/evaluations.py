"""Tasks to evaluate scores on combos."""
import pandas as pd
import yaml

import luigi
from bluepyemodel.ais_synthesis.tools.parallel import InitParallelFactory

from ...ais_synthesis.evaluators import evaluate_combos_rho
from ...ais_synthesis.utils import get_scores
from .ais_synthesis import SynthesizeAis
from .base_task import BaseTask
from .config import selectconfigs
from .morph_combos import CreateMorphCombosDF
from .utils import add_emodel, ensure_dir


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
        synth_combos_df = synth_combos_df[synth_combos_df.emodel == self.emodel]

        with InitParallelFactory(self.parallel_lib) as parallel_factory:
            synth_combos_df = evaluate_combos_rho(
                synth_combos_df,
                self.emodel_db,
                emodels=[self.emodel],
                morphology_path=self.morphology_path,
                continu=self.continu,
                parallel_factory=parallel_factory,
                save_traces=self.save_traces,
                combos_db_filename=add_emodel(self.eval_db_path, self.emodel),
            )

        megate_thresholds = yaml.safe_load(open(selectconfigs().megate_thresholds_path, "r"))
        synth_combos_with_scores_df = get_scores(synth_combos_df, megate_thresholds["ignore"])
        ensure_dir(self.output().path)
        synth_combos_with_scores_df.to_csv(self.output().path, index=False)


class EvaluateGeneric(BaseTask):
    """Evaluate the cells witht synthesized AIS."""

    emodel = luigi.Parameter()
    save_traces = luigi.BoolParameter(default=False)
    eval_db_path = luigi.Parameter(default="eval_db.sql")
    morphology_path = luigi.Parameter(default="morphology_path")

    def requires(self):
        """"""
        return CreateMorphCombosDF()

    def run(self):
        """Run."""

        morphs_combos_df = pd.read_csv(self.input().path)
        morphs_combos_df = morphs_combos_df[morphs_combos_df.emodel == self.emodel]

        with InitParallelFactory(self.parallel_lib) as parallel_factory:
            morphs_combos_df = evaluate_combos_rho(
                morphs_combos_df,
                self.emodel_db,
                emodels=[self.emodel],
                morphology_path=self.morphology_path,
                continu=self.continu,
                parallel_factory=parallel_factory,
                save_traces=self.save_traces,
                combos_db_filename=add_emodel(self.eval_db_path, self.emodel),
            )
        megate_thresholds = yaml.safe_load(open(selectconfigs().megate_thresholds_path, "r"))
        morphs_combos_with_scores_df = get_scores(morphs_combos_df, megate_thresholds["ignore"])
        ensure_dir(self.output().path)
        morphs_combos_with_scores_df.to_csv(self.output().path, index=False)


class EvaluateExemplars(BaseTask):
    """Evaluate all exemplars for select step with tresholds a la MM."""

    emodels = luigi.ListParameter(default=None)
    save_traces = luigi.BoolParameter(default=False)
    morphology_path = luigi.Parameter(default="morphology_path")
    eval_db_path = luigi.Parameter(default="eval_db.sql")

    def run(self):
        """Run."""
        morphs_combos_df = pd.read_csv(selectconfigs().exemplar_morphs_combos_df_path)

        # add emodels with seeds
        for emodel in self.emodels:
            if len(emodel.split("_")) == 3:
                _df = morphs_combos_df[morphs_combos_df.emodel == "_".join(emodel.split("_")[:2])]
                _df = _df.assign(emodel=emodel)
                morphs_combos_df = pd.concat([morphs_combos_df, _df]).reset_index(drop=True)

        morphs_combos_df = morphs_combos_df[morphs_combos_df.emodel.isin(self.emodels)]
        morphs_combos_df = morphs_combos_df[morphs_combos_df.for_optimisation == 1]

        ensure_dir(self.eval_db_path)
        with InitParallelFactory(self.parallel_lib) as parallel_factory:
            morphs_combos_df = evaluate_combos_rho(
                morphs_combos_df,
                self.emodel_db,
                emodels=None,
                morphology_path=self.morphology_path,
                continu=self.continu,
                parallel_factory=parallel_factory,
                save_traces=self.save_traces,
                combos_db_filename=self.eval_db_path,
            )

        megate_thresholds = yaml.safe_load(open(selectconfigs().megate_thresholds_path, "r"))
        morphs_combos_with_scores_df = get_scores(morphs_combos_df, megate_thresholds["ignore"])

        ensure_dir(self.output().path)
        morphs_combos_with_scores_df.to_csv(self.output().path, index=False)
