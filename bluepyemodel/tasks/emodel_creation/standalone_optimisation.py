"""This module performs optimisation only, as a standalone little luigi workflow."""
import json
from collections import defaultdict
from pathlib import Path

import deap.tools
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from luigi_tools.task import WorkflowTask as _WorkflowTask

from bluepyemodel import api
from bluepyemodel.emodel_pipeline.plotting import optimization as plot_optimization
from bluepyemodel.emodel_pipeline.plotting import scores as plot_scores
from bluepyemodel.optimisation import setup_and_run_optimisation
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.tasks.config import EmodelAPIConfig
from bluepyemodel.tasks.utils import get_mapper


# taken from ../luigi_tools to avoid pulling bbp-workflow stuff for nothing
class WorkflowTask(_WorkflowTask):
    """Workflow task with loaded emodel_db."""

    backend = luigi.Parameter(
        default="multiprocessing", config_path=dict(section="parallel", name="backend")
    )

    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

        self.emodel_db = api.get_db(EmodelAPIConfig().api, None, **EmodelAPIConfig().api_args)

    def get_mapper(self):
        """Get a mapper for parallel computations."""
        return get_mapper(self.backend)


class WorkflowWrapperTask(WorkflowTask, luigi.WrapperTask):
    """Base wrapper class with global parameters."""


class OptimiseSingleEmodel(WorkflowTask):
    """Optimise one emodel."""

    emodel = luigi.Parameter()
    seed = luigi.IntParameter(default=42)
    opt_params = luigi.DictParameter(
        default={"offspring_size": 10, "weight_hv": 0.4, "sigma": 0.4, "halloffame_size": 1000}
    )
    optimizer = luigi.Parameter(default="MO-CMA")
    max_ngen = luigi.IntParameter(default=1000)
    checkpoint_path = luigi.Parameter(default="checkpoint.pkl")
    morph_path = luigi.Parameter(default=None)

    resume_optimisation = luigi.BoolParameter(default=False)

    def run(self):
        """ """

        if self.morph_path:
            self.emodel_db.morph_path = self.morph_path
        self.emodel_db.emodel = self.emodel
        _opt_params = dict(self.opt_params)
        if "halloffame_size" in self.opt_params:
            _opt_params["hof"] = deap.tools.HallOfFame(_opt_params.pop("halloffame_size"))

        setup_and_run_optimisation(
            self.emodel_db,
            self.emodel,
            seed=self.seed,
            mapper=self.get_mapper(),
            opt_params=_opt_params,
            optimizer=self.optimizer,
            max_ngen=self.max_ngen,
            checkpoint_path=Path(self.checkpoint_path),
            continue_opt=self.resume_optimisation,
            githash="no_hash",
        )

    def output(self):
        """ """
        return luigi.LocalTarget(self.checkpoint_path)

    def complete(self):
        """ """
        if self.resume_optimisation:
            return False
        return WorkflowTask.complete(self)


class OptimiseEmodels(WorkflowTask):
    """Optimise many emodels in parallel."""

    emodels = luigi.ListParameter()
    n_seeds = luigi.IntParameter(default=10)
    seed = luigi.IntParameter(default=42)
    checkpoint_dir = luigi.Parameter(default="checkpoints")
    checkpoints_df = luigi.Parameter(default="checkpoints.csv")
    resume_optimisation = luigi.BoolParameter(default=False)

    def requires(self):
        """ """
        np.random.seed(self.seed)
        Path(self.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        tasks = {}
        for emodel in self.emodels:
            for seed in np.random.randint(1e5, size=self.n_seeds):
                checkpoint_path = str(
                    Path(self.checkpoint_dir) / f"checkpoint_{emodel}_no_hash_{seed}.pkl"
                )
                tasks[(emodel, seed)] = OptimiseSingleEmodel(
                    emodel=emodel,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                    resume_optimisation=self.resume_optimisation,
                )
        return tasks

    def run(self):
        """ """
        checkpoints = pd.DataFrame()
        for i, ((emodel, seed), checkpoint_path) in enumerate(self.input().items()):
            checkpoints.loc[i, "emodel"] = emodel
            checkpoints.loc[i, "seed"] = seed
            checkpoints.loc[i, "checkpoint_path"] = checkpoint_path.path

        checkpoints.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return luigi.LocalTarget(self.checkpoints_df)


class CreateFinal(WorkflowTask):
    """Create final.json with best emodels."""

    final_path = luigi.Parameter(default="final.json")

    def requires(self):
        """ """
        return OptimiseEmodels()

    def run(self):
        """ """
        self.emodel_db.final_path = Path(self.output().path)
        checkpoints = pd.read_csv(self.input().path)

        for gid in checkpoints.index:
            self.emodel_db.emodel = checkpoints.loc[gid, "emodel"]
            store_best_model(
                emodel_db=self.emodel_db,
                emodel=checkpoints.loc[gid, "emodel"],
                seed=int(checkpoints.loc[gid, "seed"]),
                checkpoint_path=checkpoints.loc[gid, "checkpoint_path"],
            )

    def output(self):
        """ """
        return luigi.LocalTarget(self.final_path)


class PlotEmodelOptimisations(WorkflowTask):
    """Plot results of optimisation of emodels"""

    figure_dir = luigi.Parameter(default="optimization_plots")
    all_scores_fig = luigi.Parameter(default="all_scores.pdf")
    resume_optimisation = luigi.BoolParameter(default=False)

    def requires(self):
        """ """
        return {"emodels": OptimiseEmodels(), "final": CreateFinal()}

    def run(self):
        """ """
        checkpoints = pd.read_csv(self.input()["emodels"].path)

        for gid in checkpoints.index:
            plot_optimization(
                checkpoint_path=checkpoints.loc[gid, "checkpoint_path"],
                figures_dir=self.figure_dir,
            )

        # plot the feature scores of each emodels
        final = json.load(open(self.input()["final"].path, "r"))
        all_scores = defaultdict(list)
        for name, emodel in final.items():
            emodel["scores"] = emodel["fitness"]
            all_scores[emodel["emodel"]].append(emodel["score"])
            plot_scores(emodel, figures_dir=self.figure_dir)

        # plot the histrogram of the scores per emodel (over seeds)
        plt.figure()
        for emodel, scores in all_scores.items():
            plt.hist(scores, label=emodel, histtype="step")
        plt.xlabel("score")
        plt.legend(loc="best")
        plt.savefig(self.output().path)

    def output(self):
        """ """
        return luigi.LocalTarget(self.all_scores_fig)


class CollectAllEmodels(WorkflowTask):
    """Collect all emodels from halloffame."""

    emodels_all_df = luigi.Parameter(default="all_emodels_df.csv")

    def requires(self):
        """ """
        return OptimiseEmodels()

    def run(self):
        """ """
        checkpoints = pd.read_csv(self.input().path)

        from bluepyemodel.emodel_pipeline.utils import read_checkpoint
        from bluepyemodel.evaluation.evaluation import get_evaluator_from_db

        i = 0
        for emodel, _checkpoints in checkpoints.groupby("emodel"):
            self.emodel_db.emodel = emodel
            cell_evaluator = get_evaluator_from_db(emodel=emodel, db=self.emodel_db)
            param_names = [("param", param) for param in cell_evaluator.param_names]
            feature_names = [
                ("feature", obj.name) for obj in cell_evaluator.fitness_calculator.objectives
            ]

            parameters = pd.DataFrame(
                columns=pd.MultiIndex.from_tuples(param_names + feature_names)
            )

            for _id in _checkpoints.index:
                run = read_checkpoint(_checkpoints.loc[_id, "checkpoint_path"])
                for params in run["halloffame"]:
                    parameters.loc[i, "emodel"] = emodel
                    parameters.loc[i, "fitness"] = params.fitness.reduce
                    for name, param in zip(param_names, params):
                        parameters.loc[i, name] = param
                    for name, feat in zip(feature_names, params.fitness.values):
                        parameters.loc[i, name] = feat
                    i += 1

        parameters.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return luigi.LocalTarget(self.emodels_all_df)


class RunAll(WorkflowWrapperTask):
    """Run all tasks about emodels"""

    def requires(self):
        """ """
        return [CollectAllEmodels(), PlotEmodelOptimisations(), CreateFinal()]
