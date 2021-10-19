"""Luigi tasks for emodel optimisation."""
import multiprocessing
from pathlib import Path

import luigi

from bluepyemodel.emodel_pipeline.emodel_pipeline import extract_save_features_protocols
from bluepyemodel.emodel_pipeline.plotting import optimization
from bluepyemodel.emodel_pipeline.plotting import plot_models
from bluepyemodel.optimisation import get_checkpoint_path
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.tasks.luigi_tools import IPyParallelTask
from bluepyemodel.tasks.luigi_tools import WorkflowTarget
from bluepyemodel.tasks.luigi_tools import WorkflowTask
from bluepyemodel.tasks.luigi_tools import WorkflowWrapperTask
from bluepyemodel.tools.mechanisms import compile_mechs

# pylint: disable=W0235


class EfeaturesProtocolsTarget(WorkflowTarget):
    """Target to check if efeatures and protocols are present in the database."""

    def __init__(self, emodel, ttype):
        """Constructor."""
        super().__init__(emodel=emodel, ttype=ttype)

    def exists(self):
        """Check if the features and protocols have been created."""
        _ = self.access_point.has_protocols_and_features()
        return _


class ExtractEFeatures(WorkflowTask):
    """Luigi wrapper for extract_efeatures in emodel_pipeline.EModel_pipeline.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
    """

    def run(self):
        """ """

        mapper = self.get_mapper()
        _ = extract_save_features_protocols(
            access_point=self.access_point, emodel=self.emodel, mapper=mapper
        )

    def output(self):
        """ """
        return EfeaturesProtocolsTarget(emodel=self.emodel, ttype=self.ttype)


class CompileMechanisms(WorkflowTask):
    """Luigi wrapper for optimisation.compile_mechs

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
        species (str): name of the species.
        brain_region (str): name of the brain_region.
    """

    species = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default=None)

    def run(self):
        """ """
        self.access_point.get_model_configuration()

        compile_mechs("./mechanisms")

    def output(self):
        """ """

        config = self.access_point.get_model_configuration()

        targets = []
        for mech in config.mechanism_names:
            if mech != "pas":
                targets.append(luigi.LocalTarget(Path("x86_64") / f"{mech}.c"))

        return targets


class OptimisationTarget(WorkflowTarget):
    """Target to check if an optimisation is present in the database."""

    def __init__(self, emodel, ttype, seed=1):
        """Constructor.

        Args:
           seed (int): seed used in the optimisation.
        """
        super().__init__(emodel=emodel, ttype=ttype)

        self.seed = seed

    def exists(self):
        """Check if the model is completed."""
        return self.access_point.optimisation_state(
            seed=self.seed,
            githash="",
        )


class Optimize(WorkflowTask, IPyParallelTask):
    """Luigi wrapper for emodel_pipeline.emodel_creation.optimize

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
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
        compile_mechanisms = self.access_point.pipeline_settings.compile_mechanisms

        targets = [ExtractEFeatures(emodel=self.emodel, ttype=self.ttype)]

        if compile_mechanisms:
            targets.append(
                CompileMechanisms(
                    emodel=self.emodel,
                    ttype=self.ttype,
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
        access_pt = access_point.get_db(
            args.api_from_config, args.emodel, **args.api_args_from_config
        )
        setup_and_run_optimisation(
            access_pt,
            args.emodel,
            args.seed,
            mapper=mapper,
            githash="",
        )

    def output(self):
        """ """
        return OptimisationTarget(seed=self.seed, ttype=self.ttype, emodel=self.emodel)


class BestModelTarget(WorkflowTarget):
    """Check if the best model from optimisation is present in the database."""

    def __init__(self, emodel, ttype, seed=1):
        """Constructor.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            ttype (str): name of the ttype
            seed (int): seed used in the optimisation.
        """
        super().__init__(emodel=emodel, ttype=ttype)

        self.seed = seed

    def exists(self):
        """Check if the best model is stored."""
        return self.access_point.has_best_model(seed=self.seed, githash="")


class StoreBestModels(WorkflowTask):
    """Luigi wrapper for store_best_model.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
        species (str): name of the species.
        brain_region (str): name of the brain_region.
        seed (int): seed used in the optimisation.
    """

    species = luigi.Parameter(default="")
    brain_region = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=1)

    def requires(self):
        """ """
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size

        to_run = []

        for seed in range(self.seed, self.seed + batch_size):
            to_run.append(
                Optimize(
                    emodel=self.emodel,
                    ttype=self.ttype,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=seed,
                )
            )

            to_run.append(
                PlotOptimisation(
                    emodel=self.emodel,
                    ttype=self.ttype,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=seed,
                )
            )

        to_run.append(ExtractEFeatures(emodel=self.emodel, ttype=self.ttype))

        return to_run

    def run(self):
        """ """
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        for seed in range(self.seed, self.seed + batch_size):
            # can have unfulfilled dependecies if slurm has send signal near time limit.
            if OptimisationTarget(
                emodel=self.emodel,
                ttype=self.ttype,
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
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        targets = []
        for seed in range(self.seed, self.seed + batch_size):
            targets.append(BestModelTarget(emodel=self.emodel, ttype=self.ttype, seed=seed))
        return targets


class ValidationTarget(WorkflowTarget):
    """Check if validation has been performed on the model.

    Return True if Validation task has already been performed on the model,
        even if the model is not validated.
    """

    def __init__(self, emodel, ttype, seed):
        """Constructor.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            ttype (str): name of the ttype.
            seed (int): seed used in the optimisation.
        """
        super().__init__(emodel=emodel, ttype=ttype)

        self.seed = seed

    def exists(self):
        """Check if the model is completed for all given seeds."""
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        checked_for_all_seeds = [
            self.access_point.is_checked_by_validation(seed=seed, githash="")
            for seed in range(self.seed, self.seed + batch_size)
        ]
        return all(checked_for_all_seeds)


class Validation(WorkflowTask, IPyParallelTask):
    """Luigi wrapper for validation.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
        species (str): name of the species.
        brain_region (str): name of the brain region.
        seed (int): seed used in the optimisation.
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker. Skip task if set.
    """

    species = luigi.Parameter(default="")
    brain_region = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=1)
    graceful_killer = multiprocessing.Event()

    def requires(self):
        """ """

        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation
        compile_mechanisms = self.access_point.pipeline_settings.compile_mechanisms

        to_run = [
            StoreBestModels(
                emodel=self.emodel,
                ttype=self.ttype,
                species=self.species,
                brain_region=self.brain_region,
                seed=self.seed,
            )
        ]
        if compile_mechanisms:
            to_run.append(
                CompileMechanisms(
                    emodel=self.emodel,
                    ttype=self.ttype,
                    species=self.species,
                    brain_region=self.brain_region,
                )
            )

        if plot_optimisation:
            to_run.append(
                PlotModels(
                    emodel=self.emodel,
                    ttype=self.ttype,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=self.seed,
                )
            )

        to_run.append(ExtractEFeatures(emodel=self.emodel, ttype=self.ttype))

        return to_run

    def run(self):
        """Prepare self.args, then call bbp-workflow's IPyParallelTask's run()."""
        attrs = [
            "backend",
            "emodel",
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

        args = parser.parse_args()

        # -- run validation -- #
        mapper = get_mapper(args.backend)
        access_pt = access_point.get_db(
            args.api_from_config, args.emodel, **args.api_args_from_config
        )

        validate(access_pt, args.emodel, mapper)

    def output(self):
        """ """
        return ValidationTarget(emodel=self.emodel, ttype=self.ttype, seed=self.seed)


class EModelCreationTarget(WorkflowTarget):
    """Check if the the model is validated for any seed."""

    def __init__(self, emodel, ttype):
        """Constructor.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            ttype (str): name of the ttype.
        """
        super().__init__(emodel=emodel, ttype=ttype)

    def exists(self):
        """Check if the model is completed."""

        return self.access_point.is_validated(githash="")


class EModelCreation(WorkflowTask):
    """Main Wrokflow Task. Creates an emodel.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
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
                    ttype=self.ttype,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=seed,
                )
            )
            seed += batch_size

        assert self.output().exists()

    def output(self):
        """ """
        return EModelCreationTarget(emodel=self.emodel, ttype=self.ttype)


class EModelCreationWrapper(WorkflowWrapperTask):
    """Luigi wrapper for launching EModel Creation pipeline.

    For now, launches only one EModel Creation pipeline. Could be changed later.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
        species (str): name of the species.
        brain_region (str): name of the brain region.
    """

    species = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default=None)

    def requires(self):
        """ """
        to_run = [
            EModelCreation(
                emodel=self.emodel,
                ttype=self.ttype,
                species=self.species,
                brain_region=self.brain_region,
            )
        ]

        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation

        if plot_optimisation:
            to_run.append(
                PlotValidatedDistributions(
                    emodel=self.emodel,
                    ttype=self.ttype,
                    species=self.species,
                    brain_region=self.brain_region,
                )
            )

        return to_run


class OptimizeWrapper(WorkflowWrapperTask):
    """Luigi wrapper for launching multiple seeds to optimize.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
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
    """

    species = luigi.Parameter(default="")
    brain_region = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=42)

    def requires(self):
        """ """
        return Optimize(
            emodel=self.emodel,
            ttype=self.ttype,
            species=self.species,
            brain_region=self.brain_region,
            seed=self.seed,
        )

    def run(self):
        """ """
        githash = ""
        checkpoint_path = get_checkpoint_path(self.emodel, self.seed, githash)
        optimization(
            checkpoint_path=checkpoint_path,
            figures_dir=Path("./figures") / self.emodel / "optimisation",
        )

    def output(self):
        """ """
        githash = ""
        fname = f"{get_checkpoint_path(self.emodel, self.seed, githash).stem}.pdf"
        return luigi.LocalTarget(Path("./figures") / self.emodel / "optimisation" / fname)


class PlotModels(WorkflowTask):
    """Luigi wrapper for plotting the optimisation outputs.

    Parameters:
        species (str): name of the species.
        brain_region (str): name of the brain region.
        seed (int): seed used in the optimisation.
    """

    species = luigi.Parameter(default="")
    brain_region = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=42)

    def requires(self):
        """ """
        requires = [
            ExtractEFeatures(emodel=self.emodel, ttype=self.ttype),
            StoreBestModels(
                emodel=self.emodel,
                ttype=self.ttype,
                species=self.species,
                brain_region=self.brain_region,
                seed=self.seed,
            ),
        ]

        return requires

    def run(self):
        """ """

        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        additional_protocols = self.access_point.pipeline_settings.additional_protocols

        mapper = self.get_mapper()
        plot_models(
            access_point=self.access_point,
            emodel=self.emodel,
            mapper=mapper,
            seeds=range(self.seed, self.seed + batch_size),
            githashs="",
            figures_dir=Path("./figures") / self.emodel,
            additional_protocols=additional_protocols,
            plot_distributions=plot_optimisation,
            plot_traces=plot_optimisation,
            plot_scores=plot_optimisation,
        )

    def output(self):
        """ """

        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation

        outputs = []
        if plot_optimisation:
            # distribution
            fname = f"{self.emodel}_parameters_distribution.pdf"
            fpath = Path("./figures") / self.emodel / "distributions" / "all" / fname
            outputs.append(luigi.LocalTarget(fpath))

            # scores
            for seed in range(self.seed, self.seed + batch_size):
                # change fname if githash is implemented
                fname = f"{self.emodel}_{seed}_scores.pdf"
                fpath = Path("./figures") / self.emodel / "scores" / "all" / fname
                outputs.append(luigi.LocalTarget(fpath))

            # traces
            for seed in range(self.seed, self.seed + batch_size):
                fname = f"{self.emodel}__{seed}_traces.pdf"
                fpath = Path("./figures") / self.emodel / "traces" / "all" / fname
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
            emodel=self.emodel,
            ttype=self.ttype,
            species=self.species,
            brain_region=self.brain_region,
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
