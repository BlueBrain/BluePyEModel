"""Luigi tasks for emodel optimisation."""
import multiprocessing
from pathlib import Path

import luigi
from luigi_tools.task import ParamRef
from luigi_tools.task import copy_params

from bluepyemodel.emodel_pipeline.emodel_pipeline import extract_save_features_protocols
from bluepyemodel.emodel_pipeline.plotting import optimization
from bluepyemodel.emodel_pipeline.plotting import plot_models
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.tasks.emodel_creation.config import OptimizeConfig
from bluepyemodel.tasks.luigi_tools import BoolParameterCustom
from bluepyemodel.tasks.luigi_tools import IPyParallelTask
from bluepyemodel.tasks.luigi_tools import WorkflowTarget
from bluepyemodel.tasks.luigi_tools import WorkflowTask
from bluepyemodel.tasks.luigi_tools import WorkflowWrapperTask
from bluepyemodel.tools.mechanisms import copy_and_compile_mechanisms

# pylint: disable=W0235


class EfeaturesProtocolsTarget(WorkflowTarget):
    """Target to check if efeatures and protocols are present in the database."""

    def __init__(self, emodel):
        """Constructor."""
        super().__init__(emodel=emodel)

    def exists(self):
        """Check if the features and protocols have been created."""
        _ = self.emodel_db.has_protocols_and_features()
        return _


class ExtractEFeatures(WorkflowTask):
    """Luigi wrapper for extract_efeatures in emodel_pipeline.EModel_pipeline.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        threshold_nvalue_save (int): lower bounds of the number of values required
            to save an efeature.
        name_Rin_protocol (str): name of the protocol that should be used to compute
            the input resistance. Only used when db_api is 'singlecell'
        name_rmp_protocol (str): name of the protocol that should be used to compute
            the resting membrane potential. Only used when db_api is 'singlecell'.
        validation_protocols (dict): Of the form {"ecodename": [targets]}. Only used
            when db_api is 'singlecell'.
        plot (bool): True to plot the traces and the efeatures.
    """

    species = luigi.Parameter(default=None)

    threshold_nvalue_save = luigi.IntParameter(default=1)
    name_Rin_protocol = luigi.Parameter(default=None)
    name_rmp_protocol = luigi.Parameter(default=None)
    validation_protocols = luigi.DictParameter(default=None)
    plot = BoolParameterCustom(default=False)

    def run(self):
        """ """
        mapper = self.get_mapper()
        _ = extract_save_features_protocols(
            emodel_db=self.emodel_db,
            emodel=self.emodel,
            species=self.species,
            threshold_nvalue_save=self.threshold_nvalue_save,
            mapper=mapper,
            name_Rin_protocol=self.name_Rin_protocol,
            name_rmp_protocol=self.name_rmp_protocol,
            validation_protocols=self.validation_protocols,
            plot=self.plot,
        )

    def output(self):
        """ """
        return EfeaturesProtocolsTarget(emodel=self.emodel)


class CompileMechanisms(WorkflowTask):
    """Luigi wrapper for optimisation.copy_and_compile_mechs

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        mechanisms_dir (str): path of the directory in which the mechanisms
            will be copied and/or compiled. It has to be a subdirectory of
            working_dir.
        copy_mechanisms (bool): should the mod files be copied in the local
            mechanisms_dir directory.
    """

    species = luigi.Parameter(default=None)
    mechanisms_dir = luigi.Parameter(default="mechanisms")
    copy_mechanisms = BoolParameterCustom(default=False)

    def run(self):
        """ """
        copy_and_compile_mechanisms(
            self.emodel_db,
            self.emodel,
            self.species,
            self.copy_mechanisms,
            self.mechanisms_dir,
            githash="",
        )

    def output(self):
        """ """
        return luigi.LocalTarget(Path("x86_64") / "special")


class OptimisationTarget(WorkflowTarget):
    """Target to check if an optimisation is present in the database."""

    def __init__(
        self,
        emodel,
        seed=1,
        checkpoint_dir=None,
    ):
        """Constructor.

        Args:
           seed (int): seed used in the optimisation.
           checkpoint_dir (str): path to the repo where files used as a checkpoint by BluePyOpt are.
        """
        super().__init__(emodel=emodel)

        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

    def exists(self):
        """Check if the model is completed."""
        return self.emodel_db.optimisation_state(
            self.checkpoint_dir,
            seed=self.seed,
            githash="",
        )


@copy_params(
    mechanisms_dir=ParamRef(OptimizeConfig),
    max_ngen=ParamRef(OptimizeConfig),
    stochasticity=ParamRef(OptimizeConfig),
    copy_mechanisms=ParamRef(OptimizeConfig),
    compile_mechanisms=ParamRef(OptimizeConfig),
    opt_params=ParamRef(OptimizeConfig),
    optimizer=ParamRef(OptimizeConfig),
    checkpoint_dir=ParamRef(OptimizeConfig),
    timeout=ParamRef(OptimizeConfig),
)
class Optimize(WorkflowTask, IPyParallelTask):
    """Luigi wrapper for emodel_pipeline.emodel_creation.optimize

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        seed (int): seed used in the optimisation.
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker.
            When set, will gracefully exit main loop in Optimize (in deap algorithm).
    """

    # if default not set, crashes when parameters are read by luigi_tools.copy_params
    species = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=42)
    graceful_killer = multiprocessing.Event()

    def requires(self):
        """ """
        targets = [ExtractEFeatures(emodel=self.emodel, species=self.species)]

        if self.compile_mechanisms:
            targets.append(
                CompileMechanisms(
                    emodel=self.emodel,
                    species=self.species,
                    mechanisms_dir=self.mechanisms_dir,
                    copy_mechanisms=self.copy_mechanisms,
                )
            )
        return targets

    def run(self):
        """Prepare self.args, then call bbp-workflow's IPyParallelTask's run()."""
        attrs = [
            "backend",
            "emodel",
            "seed",
            "species",
            "stochasticity",
            "timeout",
            "opt_params",
            "optimizer",
            "max_ngen",
            "checkpoint_dir",
        ]
        self.prepare_args_for_remote_script(attrs)

        super().run()

    def remote_script(self):
        """Catch arguments from parsing, and run optimisation."""
        # This function will be copied into a file, and then
        # arguments will be passed to it from command line
        # so arguments should be read using argparse.
        # All functions used should be imported.
        # Class methods and attributes cannot be used.

        # -- imports -- #
        import argparse
        import json

        from bluepyemodel import api
        from bluepyemodel.optimisation import setup_and_run_optimisation
        from bluepyemodel.tasks.utils import get_mapper

        # -- parsing -- #
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", default=None, type=str)
        parser.add_argument("--api_from_config", default="singlecell", type=str)
        parser.add_argument(
            "--api_args_from_config",
            default=", ".join(
                (
                    '{"emodel_dir": None',
                    '"recipes_path": None',
                    '"final_path": None',
                    '"legacy_dir_structure": None',
                    '"extract_config": None}',
                )
            ),
            type=json.loads,
        )
        parser.add_argument("--emodel", default=None, type=str)
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--species", default=None, type=str)
        parser.add_argument("--stochasticity", default=False, action="store_true")
        parser.add_argument("--timeout", default=600, type=int)
        parser.add_argument("--opt_params", default=None, type=json.loads)
        parser.add_argument("--optimizer", default="MO-CMA", type=str)
        parser.add_argument("--max_ngen", default=1000, type=int)
        parser.add_argument("--checkpoint_dir", default="./checkpoints/", type=str)

        args = parser.parse_args()

        # -- run optimisation -- #
        mapper = get_mapper(args.backend)
        emodel_db = api.get_db(args.api_from_config, args.emodel, **args.api_args_from_config)
        setup_and_run_optimisation(
            emodel_db,
            args.emodel,
            args.seed,
            stochasticity=args.stochasticity,
            include_validation_protocols=False,
            timeout=args.timeout,
            mapper=mapper,
            opt_params=args.opt_params,  # these should be real parameters from luigi.cfg
            optimizer=args.optimizer,
            max_ngen=args.max_ngen,
            checkpoint_dir=args.checkpoint_dir,
            githash="",
        )

    def output(self):
        """ """
        return OptimisationTarget(
            checkpoint_dir=self.checkpoint_dir, seed=self.seed, emodel=self.emodel
        )


class BestModelTarget(WorkflowTarget):
    """Check if the best model from optimisation is present in the database."""

    def __init__(
        self,
        emodel,
        seed=1,
    ):
        """Constructor.

        Args:
           emodel (str): name of the emodel. Has to match the name of the emodel
               under which the configuration data are stored.
           seed (int): seed used in the optimisation.
        """
        super().__init__(emodel=emodel)

        # self.emodel = emodel
        self.seed = seed

    def exists(self):
        """Check if the best model is stored."""
        return self.emodel_db.has_best_model(seed=self.seed, githash="")


@copy_params(
    stochasticity=ParamRef(OptimizeConfig),
    optimizer=ParamRef(OptimizeConfig),
    checkpoint_dir=ParamRef(OptimizeConfig),
)
class StoreBestModels(WorkflowTask):
    """Luigi wrapper for store_best_model.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        seed (int): seed used in the optimisation.
        batch_size (int): number of seeds to optimize at the same time before each validation.
        plot_optimisation (bool): True to launch task plotting required optimisations.
    """

    species = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=42)
    batch_size = luigi.IntParameter(default=1)
    plot_optimisation = BoolParameterCustom(default=False)

    def requires(self):
        """ """
        to_run = []

        for seed in range(self.seed, self.seed + self.batch_size):
            to_run.append(Optimize(emodel=self.emodel, species=self.species, seed=seed))
            if self.plot_optimisation:
                to_run.append(PlotOptimisation(emodel=self.emodel, species=self.species, seed=seed))
        return to_run

    def run(self):
        """ """
        for seed in range(self.seed, self.seed + self.batch_size):
            # can have unfulfilled dependecies if slurm has send signal near time limit.
            if OptimisationTarget(
                emodel=self.emodel,
                seed=seed,
                checkpoint_dir=self.checkpoint_dir,
            ).exists():
                store_best_model(
                    self.emodel_db,
                    self.emodel,
                    seed,
                    stochasticity=self.stochasticity,
                    include_validation_protocols=False,
                    optimizer=self.optimizer,
                    checkpoint_dir=self.checkpoint_dir,
                    githash="",
                )

    def output(self):
        """ """
        targets = []
        for seed in range(self.seed, self.seed + self.batch_size):
            targets.append(BestModelTarget(emodel=self.emodel, seed=seed))
        return targets


class ValidationTarget(WorkflowTarget):
    """Check if validation has been performed on the model.

    Return True if Validation task has already been performed on the model,
        even if the model is not validated.
    """

    def __init__(self, emodel, seed, batch_size):
        """Constructor.

        Args:
           emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            seed (int): seed used in the optimisation.
            batch_size (int): number of seeds to optimize at the same time before each validation.
        """
        super().__init__(emodel=emodel)

        self.seed = seed
        self.batch_size = batch_size

    def exists(self):
        """Check if the model is completed for all given seeds."""
        checked_for_all_seeds = [
            self.emodel_db.is_checked_by_validation(seed=seed, githash="")
            for seed in range(self.seed, self.seed + self.batch_size)
        ]
        return all(checked_for_all_seeds)


@copy_params(
    stochasticity=ParamRef(OptimizeConfig),
    copy_mechanisms=ParamRef(OptimizeConfig),
    compile_mechanisms=ParamRef(OptimizeConfig),
    mechanisms_dir=ParamRef(OptimizeConfig),
)
class Validation(WorkflowTask, IPyParallelTask):
    """Luigi wrapper for validation.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        seed (int): seed used in the optimisation.
        batch_size (int): number of seeds to optimize at the same time before each validation.
        additional_protocols (dict): definition of supplementary protocols. See
            examples/optimisation for usage.
        threshold (float): threshold under which the validation function returns True.
        validation_protocols_only (bool): True to only use validation protocols
            during validation.
        validation_function (str): function used to decide if a model
            passes validation or not. Should rely on emodel['scores'] and
            emodel['scores_validation']. See bluepyemodel/validation for examples.
            Should be a function name in bluepyemodel.validation.validation_functions
        plot_distributions (bool): True to plot parameters distributions of required models.
        plot_traces (bool): True to plot traces of required models.
        plot_scores (bool): True to plot scores of required models.
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker. Skip task if set.
    """

    species = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=42)
    batch_size = luigi.IntParameter(default=1)
    additional_protocols = luigi.DictParameter(default=None)
    threshold = luigi.FloatParameter(default=5.0)
    validation_protocols_only = BoolParameterCustom(default=False)
    # default should be string and not None, because
    # when this task is yielded, the default is serialized
    # and None becomes 'None'
    validation_function = luigi.Parameter(default="")
    plot_distributions = BoolParameterCustom(default=False)
    plot_traces = BoolParameterCustom(default=False)
    plot_scores = BoolParameterCustom(default=False)
    graceful_killer = multiprocessing.Event()

    def requires(self):
        """ """
        to_run = [
            StoreBestModels(
                emodel=self.emodel, species=self.species, seed=self.seed, batch_size=self.batch_size
            )
        ]
        if self.compile_mechanisms:
            to_run.append(
                CompileMechanisms(
                    emodel=self.emodel,
                    species=self.species,
                    mechanisms_dir=self.mechanisms_dir,
                    copy_mechanisms=self.copy_mechanisms,
                )
            )
        if self.plot_distributions or self.plot_traces or self.plot_scores:
            to_run.append(
                PlotModels(
                    emodel=self.emodel,
                    species=self.species,
                    seed=self.seed,
                    batch_size=self.batch_size,
                    plot_distributions=self.plot_distributions,
                    plot_traces=self.plot_traces,
                    plot_scores=self.plot_scores,
                    additional_protocols=self.additional_protocols,
                )
            )
        return to_run

    def run(self):
        """Prepare self.args, then call bbp-workflow's IPyParallelTask's run()."""
        attrs = [
            "backend",
            "emodel",
            "species",
            "stochasticity",
            "validation_function",
            "additional_protocols",
            "threshold",
            "validation_protocols_only",
        ]
        self.prepare_args_for_remote_script(attrs)

        super().run()

    def remote_script(self):
        """Catch arguments from parsing, and run validation."""
        # -- imports -- #
        import argparse
        import json

        from bluepyemodel import api
        from bluepyemodel.tasks.utils import get_mapper
        from bluepyemodel.validation.validation import validate

        # -- parsing -- #
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", default=None, type=str)
        parser.add_argument("--api_from_config", default="singlecell", type=str)
        parser.add_argument(
            "--api_args_from_config",
            default=", ".join(
                (
                    '{"emodel_dir": None',
                    '"recipes_path": None',
                    '"final_path": None',
                    '"legacy_dir_structure": None',
                    '"extract_config": None}',
                )
            ),
            type=json.loads,
        )
        parser.add_argument("--emodel", default=None, type=str)
        parser.add_argument("--species", default=None, type=str)
        parser.add_argument("--stochasticity", default=False, action="store_true")
        parser.add_argument("--validation_function", default="", type=str)
        parser.add_argument("--additional_protocols", default=None, type=json.loads)
        parser.add_argument("--threshold", default=5.0, type=float)
        parser.add_argument("--validation_protocols_only", default=False, action="store_true")

        args = parser.parse_args()

        # -- run validation -- #
        mapper = get_mapper(args.backend)
        emodel_db = api.get_db(args.api_from_config, args.emodel, **args.api_args_from_config)

        validate(
            emodel_db,
            args.emodel,
            mapper,
            validation_function=args.validation_function,
            stochasticity=args.stochasticity,
            additional_protocols=args.additional_protocols,
            threshold=args.threshold,
            validation_protocols_only=args.validation_protocols_only,
        )

    def output(self):
        """ """
        return ValidationTarget(emodel=self.emodel, seed=self.seed, batch_size=self.batch_size)


class EModelCreationTarget(WorkflowTarget):
    """Check if the the model is validated for any seed."""

    def __init__(self, emodel, n_models_to_pass_validation):
        """Constructor.

        Args:
           emodel (str): name of the emodel. Has to match the name of the emodel
               under which the configuration data are stored.
            n_models_to_pass_validation (int): minimum number of models to pass validation
                to consider the task as validated.
        """
        super().__init__(emodel=emodel)

        self.n_models_to_pass_validation = n_models_to_pass_validation

    def exists(self):
        """Check if the model is completed."""
        return self.emodel_db.is_validated(
            githash="",
            n_models_to_pass_validation=self.n_models_to_pass_validation,
        )


class EModelCreation(WorkflowTask):
    """Main Wrokflow Task. Creates an emodel.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        seed (int): seed used in the optimisation.
        batch_size (int): number of seeds to optimize at the same time before each validation.
        max_n_batch (int): maximum number of batches. Used only if limit_batches is True.
        limit_batches (bool): whether to limit the number of batches or not.
        n_models_to_pass_validation (int): minimum number of models to pass validation
            to consider the task as validated.
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker. Exit loop if set.
    """

    species = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=1)
    batch_size = luigi.IntParameter(default=10)
    max_n_batch = luigi.IntParameter(default=10)
    limit_batches = BoolParameterCustom(default=True)
    n_models_to_pass_validation = luigi.IntParameter(default=1)
    graceful_killer = multiprocessing.Event()

    def run(self):
        """Optimize e-models by batches of 10 until one is validated."""
        seed = self.seed

        while not self.output().exists() and not self.graceful_killer.is_set():
            # limit the number of batch
            if self.limit_batches and seed > self.seed + self.max_n_batch * self.batch_size:
                break

            yield (
                Validation(
                    emodel=self.emodel,
                    species=self.species,
                    seed=seed,
                    batch_size=self.batch_size,
                )
            )
            seed += self.batch_size

        assert self.output().exists()

    def output(self):
        """ """
        return EModelCreationTarget(
            emodel=self.emodel,
            n_models_to_pass_validation=self.n_models_to_pass_validation,
        )


class EModelCreationWrapper(WorkflowWrapperTask):
    """Luigi wrapper for launching EModel Creation pipeline.

    For now, launches only one EModel Creation pipleine. Could be changed later.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        plot_validated_distributions (bool): True to plot the parameters distributions
            of all the validated models (of type self.emodel, self.species).
    """

    species = luigi.Parameter(default=None)
    plot_validated_distributions = BoolParameterCustom(default=False)

    def requires(self):
        """ """
        to_run = [EModelCreation(emodel=self.emodel, species=self.species)]
        if self.plot_validated_distributions:
            to_run.append(PlotValidatedDistributions(emodel=self.emodel, species=self.species))
        return to_run


class OptimizeWrapper(WorkflowWrapperTask):
    """Luigi wrapper for launching multiple seeds to optimize.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        seed (int): seed used in the optimisation.
        batch_size (int): number of seeds to optimize at the same time before each validation.
    """

    species = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=42)
    batch_size = luigi.IntParameter(default=10)

    def requires(self):
        """ """
        to_run = []
        for seed in range(self.seed, self.seed + self.batch_size):
            to_run.append(Optimize(emodel=self.emodel, species=self.species, seed=seed))
        return to_run


class PlotOptimisation(WorkflowTask):
    """Luigi wrapper for plotting the optimisation outputs.

    Parameters:
        species (str): name of the species.
        seed (int): seed used in the optimisation.
        checkpoint_dir (str): path to the repo where files used as a checkpoint by BluePyOpt are.
        figures_dir (str): path to figures repo.
    """

    species = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=42)
    checkpoint_dir = luigi.Parameter("./checkpoints/")
    figures_dir = luigi.Parameter(default="./figures")

    def requires(self):
        """ """
        return Optimize(emodel=self.emodel, species=self.species, seed=self.seed)

    def run(self):
        """ """
        githash = ""
        checkpoint_path = (
            Path(self.checkpoint_dir) / f"checkpoint__{self.emodel}__{githash}__{self.seed}.pkl"
        )
        optimization(
            checkpoint_path=checkpoint_path,
            figures_dir=Path(self.figures_dir) / self.emodel / "optimisation",
        )

    def output(self):
        """ """
        githash = ""
        fname = f"checkpoint__{self.emodel}__{githash}__{self.seed}.pdf"
        return luigi.LocalTarget(Path(self.figures_dir) / self.emodel / "optimisation" / fname)


class PlotModels(WorkflowTask):
    """Luigi wrapper for plotting the optimisation outputs.

    Parameters:
        species (str): name of the species.
        seed (int): seed used in the optimisation.
        batch_size (int): number of seeds to optimize at the same time before each validation.
        additional_protocols (dict): definition of supplementary protocols. See
            examples/optimisation for usage.
        plot_distributions (bool): True to plot parameters distributions of required models.
        plot_traces (bool): True to plot traces of required models.
        plot_scores (bool): True to plot scores of required models.
        figures_dir (str): path to figures repo.
    """

    species = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=42)
    batch_size = luigi.IntParameter(default=1)
    additional_protocols = luigi.DictParameter(default=None)
    plot_distributions = BoolParameterCustom(default=False)
    plot_traces = BoolParameterCustom(default=False)
    plot_scores = BoolParameterCustom(default=False)
    figures_dir = luigi.Parameter(default="./figures")

    def requires(self):
        """ """
        return StoreBestModels(
            emodel=self.emodel, species=self.species, seed=self.seed, batch_size=self.batch_size
        )

    def run(self):
        """ """
        mapper = self.get_mapper()
        plot_models(
            emodel_db=self.emodel_db,
            emodel=self.emodel,
            mapper=mapper,
            seeds=range(self.seed, self.seed + self.batch_size),
            figures_dir=Path(self.figures_dir) / self.emodel,
            additional_protocols=self.additional_protocols,
            plot_distributions=self.plot_distributions,
            plot_traces=self.plot_traces,
            plot_scores=self.plot_scores,
            only_validated=False,
        )

    def output(self):
        """ """
        outputs = []
        if self.plot_distributions:
            fname = f"{self.emodel}_parameters_distribution.pdf"
            fpath = Path(self.figures_dir) / self.emodel / "distributions" / "all" / fname
            outputs.append(luigi.LocalTarget(fpath))

        if self.plot_scores:
            for seed in range(self.seed, self.seed + self.batch_size):
                # change fname if githash is implemented
                fname = "{}_{}_scores.pdf".format(self.emodel, seed)
                fpath = Path(self.figures_dir) / self.emodel / "scores" / "all" / fname
                outputs.append(luigi.LocalTarget(fpath))

        if self.plot_traces:
            for seed in range(self.seed, self.seed + self.batch_size):
                fname = "{}_{}_{}_traces.pdf".format(
                    self.emodel,
                    "",  # githash
                    seed,
                )
                fpath = Path(self.figures_dir) / self.emodel / "traces" / "all" / fname
                outputs.append(luigi.LocalTarget(fpath))

        return outputs


class PlotValidatedDistributions(WorkflowTask):
    """Luigi wrapper for plotting the optimisation outputs.

    Parameters:
        species (str): name of the species.
        figures_dir (str): path to figures repo.
    """

    species = luigi.Parameter(default="")
    figures_dir = luigi.Parameter(default="./figures")

    def requires(self):
        """ """
        return EModelCreation(emodel=self.emodel, species=self.species)

    def run(self):
        """ """
        mapper = self.get_mapper()
        plot_models(
            emodel_db=self.emodel_db,
            emodel=self.emodel,
            mapper=mapper,
            figures_dir=Path(self.figures_dir) / self.emodel,
            plot_distributions=True,
            plot_traces=False,
            plot_scores=False,
            only_validated=True,
        )

    def output(self):
        """ """
        fname = f"{self.emodel}_parameters_distribution.pdf"
        fpath = Path(self.figures_dir) / self.emodel / "distributions" / "validated" / fname

        return luigi.LocalTarget(fpath)
