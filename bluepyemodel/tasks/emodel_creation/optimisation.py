"""Luigi tasks for emodel optimisation."""
import multiprocessing
from pathlib import Path

import luigi

from bluepyemodel.emodel_pipeline.emodel_pipeline import extract_save_features_protocols
from bluepyemodel.emodel_pipeline.plotting import optimization
from bluepyemodel.emodel_pipeline.plotting import plot_models
from bluepyemodel.optimisation import store_best_model
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
        _ = self.access_point.has_protocols_and_features()
        return _


class ExtractEFeatures(WorkflowTask):
    """Luigi wrapper for extract_efeatures in emodel_pipeline.EModel_pipeline.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        name_Rin_protocol (str): name of the protocol that should be used to compute
            the input resistance. Only used when db_api is 'local'
        name_rmp_protocol (str): name of the protocol that should be used to compute
            the resting membrane potential. Only used when db_api is 'local'.
        validation_protocols (dict): Of the form {"ecodename": [targets]}. Only used
            when db_api is 'local'.
    """

    name_Rin_protocol = luigi.Parameter(default=None)
    name_rmp_protocol = luigi.Parameter(default=None)
    validation_protocols = luigi.DictParameter(default=None)

    def run(self):
        """ """

        mapper = self.get_mapper()
        _ = extract_save_features_protocols(
            access_point=self.access_point, emodel=self.emodel, mapper=mapper
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
        brain_region (str): name of the brain_region.
        mechanisms_dir (str): path of the directory in which the mechanisms
            will be copied and/or compiled. It has to be a subdirectory of
            working_dir.
    """

    species = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default=None)
    mechanisms_dir = luigi.Parameter(default="mechanisms")

    def run(self):
        """ """
        copy_and_compile_mechanisms(self.emodel_db)

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
        return self.access_point.optimisation_state(
            self.checkpoint_dir,
            seed=self.seed,
            githash="",
        )


class Optimize(WorkflowTask, IPyParallelTask):
    """Luigi wrapper for emodel_pipeline.emodel_creation.optimize

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        brain_region (str): name of the brain_region.
        seed (int): seed used in the optimisation.
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker.
            When set, will gracefully exit main loop in Optimize (in deap algorithm).
    """

    # if default not set, crashes when parameters are read by luigi_tools.copy_params
    species = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=42)
    graceful_killer = multiprocessing.Event()

    def requires(self):
        """ """
        targets = [ExtractEFeatures(emodel=self.emodel)]

        if self.compile_mechanisms:
            targets.append(
                CompileMechanisms(
                    emodel=self.emodel,
                    species=self.species,
                    brain_region=self.brain_region,
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
            "brain_region",
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

        from bluepyemodel import access_point
        from bluepyemodel.optimisation import setup_and_run_optimisation
        from bluepyemodel.tasks.utils import get_mapper

        # -- parsing -- #
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", default=None, type=str)
        parser.add_argument("--api_from_config", default="local", type=str)
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
        parser.add_argument("--seed", default=1, type=int)
        parser.add_argument("--species", default=None, type=str)
        parser.add_argument("--brain_region", default="", type=str)

        args = parser.parse_args()

        # -- run optimisation -- #
        mapper = get_mapper(args.backend)
        access_point = access_point.get_db(
            args.api_from_config, args.emodel, **args.api_args_from_config
        )
        setup_and_run_optimisation(
            access_point,
            args.emodel,
            args.seed,
            mapper=mapper,
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
        return self.access_point.has_best_model(seed=self.seed, githash="")


class StoreBestModels(WorkflowTask):
    """Luigi wrapper for store_best_model.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        brain_region (str): name of the brain_region.
        seed (int): seed used in the optimisation.
        batch_size (int): number of seeds to optimize at the same time before each validation.
    """

    species = luigi.Parameter(default="")
    brain_region = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=1)
    batch_size = luigi.IntParameter(default=3)

    def requires(self):
        """ """
        to_run = []

        for seed in range(self.seed, self.seed + self.batch_size):
            to_run.append(
                Optimize(
                    emodel=self.emodel,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=seed,
                )
            )

            to_run.append(
                PlotOptimisation(
                    emodel=self.emodel,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=seed,
                )
            )

        to_run.append(ExtractEFeatures(emodel=self.emodel))

        return to_run

    def run(self):
        """ """
        for seed in range(self.seed, self.seed + self.batch_size):
            # can have unfulfilled dependecies if slurm has send signal near time limit.
            if OptimisationTarget(
                emodel=self.emodel,
                seed=seed,
            ).exists():
                store_best_model(
                    self.access_point,
                    self.emodel,
                    seed,
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
            self.access_point.is_checked_by_validation(seed=seed, githash="")
            for seed in range(self.seed, self.seed + self.batch_size)
        ]
        return all(checked_for_all_seeds)


class Validation(WorkflowTask, IPyParallelTask):
    """Luigi wrapper for validation.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        brain_region (str): name of the brain region.
        seed (int): seed used in the optimisation.
        additional_protocols (dict): definition of supplementary protocols. See
            examples/optimisation for usage.
        validation_protocols_only (bool): True to only use validation protocols
            during validation.
        validation_function (str): function used to decide if a model
            passes validation or not. Should rely on emodel['scores'] and
            emodel['scores_validation']. See bluepyemodel/validation for examples.
            Should be a function name in bluepyemodel.validation.validation_functions
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker. Skip task if set.
    """

    species = luigi.Parameter(default="")
    brain_region = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=1)
    additional_protocols = luigi.DictParameter(default=None)
    validation_protocols_only = BoolParameterCustom(default=False)
    # default should be string and not None, because
    # when this task is yielded, the default is serialized
    # and None becomes 'None'
    validation_function = luigi.Parameter(default="")
    graceful_killer = multiprocessing.Event()

    def requires(self):
        """ """

        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation

        to_run = [
            StoreBestModels(
                emodel=self.emodel,
                species=self.species,
                brain_region=self.brain_region,
                seed=self.seed,
                batch_size=batch_size,
            )
        ]
        if self.compile_mechanisms:
            to_run.append(
                CompileMechanisms(
                    emodel=self.emodel,
                    species=self.species,
                    brain_region=self.brain_region,
                )
            )

        if plot_optimisation:
            to_run.append(
                PlotModels(
                    emodel=self.emodel,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=self.seed,
                    batch_size=batch_size,
                    plot_distributions=plot_optimisation,
                    plot_traces=plot_optimisation,
                    plot_scores=plot_optimisation,
                    additional_protocols=self.additional_protocols,
                )
            )

        to_run.append(ExtractEFeatures(emodel=self.emodel))

        return to_run

    def run(self):
        """Prepare self.args, then call bbp-workflow's IPyParallelTask's run()."""
        attrs = [
            "backend",
            "emodel",
            "species",
            "brain_region",
            "validation_function",
            "additional_protocols",
            "validation_protocols_only",
        ]
        self.prepare_args_for_remote_script(attrs)

        super().run()

    def remote_script(self):
        """Catch arguments from parsing, and run validation."""
        # -- imports -- #
        import argparse
        import json

        from bluepyemodel import access_point
        from bluepyemodel.tasks.utils import get_mapper
        from bluepyemodel.validation.validation import validate

        # -- parsing -- #
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", default=None, type=str)
        parser.add_argument("--api_from_config", default="local", type=str)
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
        parser.add_argument("--brain_region", default="", type=str)

        args = parser.parse_args()

        # -- run validation -- #
        mapper = get_mapper(args.backend)
        access_point = access_point.get_db(
            args.api_from_config, args.emodel, **args.api_args_from_config
        )

        validate(access_point, args.emodel, mapper)

    def output(self):
        """ """
        return ValidationTarget(emodel=self.emodel, seed=self.seed, batch_size=self.batch_size)


class EModelCreationTarget(WorkflowTarget):
    """Check if the the model is validated for any seed."""

    def __init__(self, emodel):
        """Constructor.

        Args:
           emodel (str): name of the emodel. Has to match the name of the emodel
               under which the configuration data are stored.
        """
        super().__init__(emodel=emodel)

    def exists(self):
        """Check if the model is completed."""

        return self.access_point.is_validated(githash="")


class EModelCreation(WorkflowTask):
    """Main Wrokflow Task. Creates an emodel.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        brain_region (str): name of the brain region.
        seed (int): seed used in the optimisation.
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker. Exit loop if set.
    """

    species = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=1)
    graceful_killer = multiprocessing.Event()

    def run(self):
        """Optimize e-models by batches of batch_size until one is validated."""
        seed = self.seed

        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        max_n_batch = self.access_point.pipeline_settings.max_n_batch

        while not self.output().exists() and not self.graceful_killer.is_set():
            # limit the number of batch
            if seed > self.seed + max_n_batch * batch_size:
                break

            yield (
                Validation(
                    emodel=self.emodel,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=seed,
                )
            )
            seed += batch_size

        assert self.output().exists()

    def output(self):
        """ """
        return EModelCreationTarget(emodel=self.emodel)


class EModelCreationWrapper(WorkflowWrapperTask):
    """Luigi wrapper for launching EModel Creation pipeline.

    For now, launches only one EModel Creation pipleine. Could be changed later.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        brain_region (str): name of the brain region.
    """

    species = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default=None)

    def requires(self):
        """ """
        to_run = [
            EModelCreation(emodel=self.emodel, species=self.species, brain_region=self.brain_region)
        ]

        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation

        if plot_optimisation:
            to_run.append(
                PlotValidatedDistributions(
                    emodel=self.emodel, species=self.species, brain_region=self.brain_region
                )
            )

        return to_run


class OptimizeWrapper(WorkflowWrapperTask):
    """Luigi wrapper for launching multiple seeds to optimize.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        brain_region (str): name of the brain region.
        seed (int): seed used in the optimisation.
        batch_size (int): number of seeds to optimize at the same time before each validation.
    """

    species = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=42)

    def requires(self):
        """ """
        to_run = []

        batch_size = self.access_point.pipeline_settings.optimisation_batch_size

        for seed in range(self.seed, self.seed + batch_size):
            to_run.append(
                Optimize(
                    emodel=self.emodel,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=seed,
                )
            )
        return to_run


class PlotOptimisation(WorkflowTask):
    """Luigi wrapper for plotting the optimisation outputs.

    Parameters:
        species (str): name of the species.
        brain_region (str): name of the brain region.
        seed (int): seed used in the optimisation.
        checkpoint_dir (str): path to the repo where files used as a checkpoint by BluePyOpt are.
        figures_dir (str): path to figures repo.
    """

    species = luigi.Parameter(default="")
    brain_region = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=42)

    def requires(self):
        """ """
        return Optimize(
            emodel=self.emodel, species=self.species, brain_region=self.brain_region, seed=self.seed
        )

    def run(self):
        """ """
        githash = ""
        checkpoint_path = (
            Path("./checkpoints/") / f"checkpoint__{self.emodel}__{githash}__{self.seed}.pkl"
        )
        optimization(
            checkpoint_path=checkpoint_path,
            figures_dir=Path("./figures") / self.emodel / "optimisation",
        )

    def output(self):
        """ """
        githash = ""
        fname = f"checkpoint__{self.emodel}__{githash}__{self.seed}.pdf"
        return luigi.LocalTarget(Path("./figures") / self.emodel / "optimisation" / fname)


class PlotModels(WorkflowTask):
    """Luigi wrapper for plotting the optimisation outputs.

    Parameters:
        species (str): name of the species.
        brain_region (str): name of the brain region.
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
    brain_region = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=42)
    additional_protocols = luigi.DictParameter(default=None)

    def requires(self):
        """ """

        batch_size = self.access_point.pipeline_settings.optimisation_batch_size

        requires = [
            ExtractEFeatures(emodel=self.emodel),
            StoreBestModels(
                emodel=self.emodel,
                species=self.species,
                brain_region=self.brain_region,
                seed=self.seed,
                batch_size=batch_size,
            ),
        ]

        return requires

    def run(self):
        """ """

        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation

        mapper = self.get_mapper()
        plot_models(
            access_point=self.access_point,
            emodel=self.emodel,
            mapper=mapper,
            seeds=range(self.seed, self.seed + self.batch_size),
            figures_dir=Path(self.figures_dir) / self.emodel,
            additional_protocols=self.additional_protocols,
            plot_distributions=plot_optimisation,
            plot_traces=plot_optimisation,
            plot_scores=plot_optimisation,
            only_validated=False,
        )

    def output(self):
        """ """

        batch_size = self.access_point.pipeline_settings.optimisation_batch_size

        outputs = []
        if self.plot_distributions:
            fname = f"{self.emodel}_parameters_distribution.pdf"
            fpath = Path(self.figures_dir) / self.emodel / "distributions" / "all" / fname
            outputs.append(luigi.LocalTarget(fpath))

        if self.plot_scores:
            for seed in range(self.seed, self.seed + batch_size):
                # change fname if githash is implemented
                fname = "{}_{}_scores.pdf".format(self.emodel, seed)
                fpath = Path(self.figures_dir) / self.emodel / "scores" / "all" / fname
                outputs.append(luigi.LocalTarget(fpath))

        if self.plot_traces:
            for seed in range(self.seed, self.seed + batch_size):
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
        brain_region (str): name of the brain region.
    """

    species = luigi.Parameter(default="")
    brain_region = luigi.Parameter(default="")

    def requires(self):
        """ """
        return EModelCreation(
            emodel=self.emodel, species=self.species, brain_region=self.brain_region
        )

    def run(self):
        """ """
        mapper = self.get_mapper()
        plot_models(
            access_point=self.access_point,
            emodel=self.emodel,
            mapper=mapper,
            figures_dir=Path("./figures") / self.emodel,
            plot_distributions=True,
            plot_traces=False,
            plot_scores=False,
            only_validated=True,
        )

    def output(self):
        """ """
        fname = f"{self.emodel}_parameters_distribution.pdf"
        fpath = Path("./figures") / self.emodel / "distributions" / "validated" / fname

        return luigi.LocalTarget(fpath)
