"""Tasks to evaluate scores on combos."""
import luigi
import pandas as pd
import yaml
from bluepyparallel import init_parallel_factory

from bluepyemodel.generalisation.evaluators import evaluate_rho
from bluepyemodel.generalisation.evaluators import evaluate_rho_axon
from bluepyemodel.generalisation.utils import get_scores
from bluepyemodel.tasks.generalisation.ais_synthesis import SynthesizeAisSoma
from bluepyemodel.tasks.generalisation.base_task import BaseTask
from bluepyemodel.tasks.generalisation.config import EvaluationLocalTarget
from bluepyemodel.tasks.generalisation.config import SelectConfig
from bluepyemodel.tasks.generalisation.morph_combos import CreateMorphCombosDF
from bluepyemodel.tasks.generalisation.utils import ensure_dir
from bluepyemodel.tools.misc_evaluators import feature_evaluation


class EvaluateSynthesis(BaseTask):
    """Evaluate the cells witht synthesized AIS."""

    emodel = luigi.Parameter()
    save_traces = luigi.BoolParameter(default=False)
    eval_db_path = luigi.Parameter(default="eval_db.sql")
    morphology_path = luigi.Parameter(default="morphology_path")
    target_path = luigi.Parameter(default="synth_combos_with_scores_df.csv")

    def requires(self):
        """Requires."""
        return {"synth_ais": SynthesizeAisSoma(emodel=self.emodel)}

    def run(self):
        """Run."""
        synth_combos_df = pd.read_csv(self.input()["synth_ais"].path)
        if "rin_no_axon" in synth_combos_df.columns:
            synth_combos_df = synth_combos_df.drop(columns=["rin_no_axon"])
        if "rin_no_soma" in synth_combos_df.columns:
            synth_combos_df = synth_combos_df.drop(columns=["rin_no_soma"])
        if "exception" in synth_combos_df.columns:
            synth_combos_df = synth_combos_df.drop(columns=["exception"])
        synth_combos_df = synth_combos_df[synth_combos_df.emodel == self.emodel]

        eval_db_path = self.set_tmp(self.add_emodel(self.eval_db_path))
        ensure_dir(eval_db_path)
        parallel_factory = init_parallel_factory(self.parallel_lib)
        synth_combos_df = evaluate_rho(
            synth_combos_df,
            self.emodel_db,
            morphology_path=self.morphology_path,
            resume=self.resume,
            parallel_factory=parallel_factory,
            db_url=eval_db_path,
        )
        synth_combos_df = evaluate_rho_axon(
            synth_combos_df,
            self.emodel_db,
            morphology_path=self.morphology_path,
            resume=self.resume,
            parallel_factory=parallel_factory,
            db_url=eval_db_path,
        )

        synth_combos_df = feature_evaluation(

        synth_combos_df = evaluate_combos_rho(
>>>>>>> dc7521c (more update)
            synth_combos_df,
            self.emodel_db,
            morphology_path=self.morphology_path,
            resume=self.resume,
            parallel_factory=parallel_factory,
            db_url=eval_db_path,
        )

        with open(SelectConfig().megate_thresholds_path, "r") as megate_thres_file:
            megate_thresholds = yaml.safe_load(megate_thres_file)
        synth_combos_with_scores_df = get_scores(synth_combos_df, megate_thresholds["ignore"])
        ensure_dir(self.output().path)
        synth_combos_with_scores_df.to_csv(self.output().path, index=False)

        parallel_factory.shutdown()

    def output(self):
        """ """
        return EvaluationLocalTarget(self.add_emodel(self.target_path))


class EvaluateGeneric(BaseTask):
    """Evaluate the cells witht synthesized AIS."""

    emodel = luigi.Parameter()
    save_traces = luigi.BoolParameter(default=False)
    eval_db_path = luigi.Parameter(default="eval_db.sql")
    morphology_path = luigi.Parameter(default="morphology_path")
    target_path = luigi.Parameter(default="generic_combos_with_scores_df.csv")

    def requires(self):
        """ """
        return CreateMorphCombosDF()

    def run(self):
        """Run."""

        morphs_combos_df = pd.read_csv(self.input().path)
        morphs_combos_df = morphs_combos_df[morphs_combos_df.emodel == self.emodel]

        eval_db_path = self.set_tmp(self.add_emodel(self.eval_db_path))
        ensure_dir(eval_db_path)
        parallel_factory = init_parallel_factory(self.parallel_lib)
        morphs_combos_df = feature_evaluation(
            morphs_combos_df,
            self.emodel_db,
            morphology_path=self.morphology_path,
            resume=self.resume,
            parallel_factory=parallel_factory,
            db_url=eval_db_path,
        )
        with open(SelectConfig().megate_thresholds_path, "r") as megate_thres_file:
            megate_thresholds = yaml.safe_load(megate_thres_file)
        morphs_combos_with_scores_df = get_scores(morphs_combos_df, megate_thresholds["ignore"])
        ensure_dir(self.output().path)
        morphs_combos_with_scores_df.to_csv(self.output().path, index=False)

        parallel_factory.shutdown()

    def output(self):
        """ """
        return EvaluationLocalTarget(self.add_emodel(self.target_path))


class EvaluateExemplars(BaseTask):
    """Evaluate all exemplars for select step with tresholds a la MM."""

    emodel = luigi.Parameter(default=None)
    save_traces = luigi.BoolParameter(default=False)
    morphology_path = luigi.Parameter(default="morphology_path")
    eval_db_path = luigi.Parameter(default="eval_db.sql")
    target_path = luigi.Parameter(default="exemplar_evaluations.csv")

    def requires(self):
        """ """
        return CreateMorphCombosDF()

    def run(self):
        """Run."""
        morphs_combos_df = pd.read_csv(self.input().path)

        # pylint: disable=no-member
        # add emodels with seeds
        if len(self.emodel.split("_")) == 3:
            _df = morphs_combos_df[morphs_combos_df.emodel == "_".join(self.emodel.split("_")[:2])]
            _df = _df.assign(emodel=self.emodel)
            morphs_combos_df = pd.concat([morphs_combos_df, _df]).reset_index(drop=True)

        morphs_combos_df = morphs_combos_df[morphs_combos_df.emodel == self.emodel]
        morphs_combos_df = morphs_combos_df[morphs_combos_df.for_optimisation == 1]

        ensure_dir(self.set_tmp(self.add_emodel(self.eval_db_path)))
        parallel_factory = init_parallel_factory(self.parallel_lib)

        morphs_combos_df = evaluate_rho(
            morphs_combos_df,
            self.emodel_db,
            morphology_path=self.morphology_path,
            resume=self.resume,
            parallel_factory=parallel_factory,
            db_url=self.set_tmp(self.add_emodel(self.eval_db_path)),
        )

        morphs_combos_df = feature_evaluation(
            morphs_combos_df,
            self.emodel_db,
            morphology_path=self.morphology_path,
            resume=self.resume,
            parallel_factory=parallel_factory,
            db_url=self.set_tmp(self.add_emodel(self.eval_db_path)),
        )

        with open(SelectConfig().megate_thresholds_path, "r") as megate_thres_file:
            megate_thresholds = yaml.safe_load(megate_thres_file)
        morphs_combos_with_scores_df = get_scores(morphs_combos_df, megate_thresholds["ignore"])

        ensure_dir(self.output().path)
        morphs_combos_with_scores_df.to_csv(self.output().path, index=False)
        parallel_factory.shutdown()

    def output(self):
        """ """
        return EvaluationLocalTarget(self.add_emodel(self.target_path))
