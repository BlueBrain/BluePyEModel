"""Luigi tasks for emodel optimisation."""
import glob
import multiprocessing
from pathlib import Path

import luigi

from bluepyemodel.emodel_pipeline.emodel_pipeline import extract_save_features_protocols
from bluepyemodel.emodel_pipeline.plotting import optimization
from bluepyemodel.emodel_pipeline.plotting import plot_models
from bluepyemodel.emodel_pipeline.utils import run_metadata_as_string
from bluepyemodel.optimisation import get_checkpoint_path
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.tasks.luigi_tools import IPyParallelTask
from bluepyemodel.tasks.luigi_tools import WorkflowTarget
from bluepyemodel.tasks.luigi_tools import WorkflowTask
from bluepyemodel.tasks.luigi_tools import WorkflowTaskRequiringMechanisms
from bluepyemodel.tasks.luigi_tools import WorkflowWrapperTask
from bluepyemodel.tools.mechanisms import compile_mechs

# pylint: disable=W0235,W0621,W0404,W0611


def _reformat_ttype(ttype):
    """ """

    if isinstance(ttype, str):
        return ttype.replace("__", " ")

    return None


class EfeaturesProtocolsTarget(WorkflowTarget):
    """Target to check if efeatures and protocols are present in the database."""

    def __init__(self, emodel, ttype, iteration_tag):
        """Constructor."""
        super().__init__(emodel=emodel, ttype=ttype, iteration_tag=iteration_tag)

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
        iteration_tag (str): tag of the current iteration
    """

    def run(self):
        """ """

        mapper = self.get_mapper()
        _ = extract_save_features_protocols(
            access_point=self.access_point, emodel=self.emodel, mapper=mapper
        )

    def output(self):
        """ """
        return EfeaturesProtocolsTarget(
            emodel=self.emodel, ttype=self.ttype, iteration_tag=self.iteration_tag
        )


class CompileMechanisms(WorkflowTaskRequiringMechanisms):
    """Luigi wrapper for optimisation.compile_mechs

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
        iteration_tag (str): tag of the current iteration
        species (str): name of the species.
        brain_region (str): name of the brain_region.
    """

    species = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default=None)

    def run(self):
        """ """
        compile_mechs("./mechanisms")

    def output(self):
        """ """
        targets = []
        for filepath in glob.glob("./mechanisms/*.mod"):
            targets.append(luigi.LocalTarget(f"./x86_64/{Path(filepath).stem}.c"))
        return targets


class OptimisationTarget(WorkflowTarget):
    """Target to check if an optimisation is present in the database."""

    def __init__(self, emodel, ttype, iteration_tag, seed=1, continue_opt=False):
        """Constructor.

        Args:
           seed (int): seed used in the optimisation.
           continue_opt (bool): whether to continue optimisation or not
                when the optimisation is not complete.
        """
        super().__init__(emodel=emodel, ttype=ttype, iteration_tag=iteration_tag)

        self.seed = seed
        self.continue_opt = continue_opt

    def exists(self):
        """Check if the model is completed."""
        return self.access_point.optimisation_state(seed=self.seed, continue_opt=self.continue_opt)


class Optimize(WorkflowTaskRequiringMechanisms, IPyParallelTask):
    """Luigi wrapper for emodel_pipeline.emodel_creation.optimize

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
        iteration_tag (str): tag of the current iteration
        species (str): name of the species.
        brain_region (str): name of the brain_region.
        seed (int): seed used in the optimisation.
        continue_unfinished_optimisation (bool): whether to continue optimisation or not
                when the optimisation is not complete.
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker.
            When set, will gracefully exit main loop in Optimize (in deap algorithm).
    """

    # if default not set, crashes when parameters are read by luigi_tools.copy_params
    species = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=42)
    continue_unfinished_optimisation = luigi.BoolParameter(default=False)
    graceful_killer = multiprocessing.Event()

    def requires(self):
        """ """
        compile_mechanisms = self.access_point.pipeline_settings.compile_mechanisms

        targets = [
            ExtractEFeatures(emodel=self.emodel, ttype=self.ttype, iteration_tag=self.iteration_tag)
        ]

        if compile_mechanisms:
            targets.append(
                CompileMechanisms(
                    emodel=self.emodel,
                    ttype=self.ttype,
                    iteration_tag=self.iteration_tag,
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
            "ttype",
            "iteration_tag",
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
        from bluepyemodel.tools.multiprocessing import get_mapper

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
        parser.add_argument("--ttype", default=None, type=str)
        parser.add_argument("--iteration_tag", default=None, type=str)
        parser.add_argument("--species", default=None, type=str)
        parser.add_argument("--brain_region", default="", type=str)

        args = parser.parse_args()

        # -- run optimisation -- #
        mapper = get_mapper(args.backend)
        access_pt = access_point.get_access_point(
            access_point=args.api_from_config,
            emodel=args.emodel,
            ttype=args.ttype,
            iteration_tag=args.iteration_tag,
            **args.api_args_from_config,
        )
        setup_and_run_optimisation(access_pt, args.seed, mapper=mapper)

    def output(self):
        """ """
        return OptimisationTarget(
            seed=self.seed,
            ttype=self.ttype,
            emodel=self.emodel,
            iteration_tag=self.iteration_tag,
            continue_opt=self.continue_unfinished_optimisation,
        )


class BestModelTarget(WorkflowTarget):
    """Check if the best model from optimisation is present in the database."""

    def __init__(self, emodel, ttype, iteration_tag, seed=1):
        """Constructor.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            ttype (str): name of the ttype
            seed (int): seed used in the optimisation.
        """
        super().__init__(emodel=emodel, ttype=ttype, iteration_tag=iteration_tag)

        self.seed = seed

    def exists(self):
        """Check if the best model is stored."""
        return self.access_point.has_best_model(seed=self.seed)


class StoreBestModels(WorkflowTaskRequiringMechanisms):
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
                    iteration_tag=self.iteration_tag,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=seed,
                )
            )

            to_run.append(
                PlotOptimisation(
                    emodel=self.emodel,
                    ttype=self.ttype,
                    iteration_tag=self.iteration_tag,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=seed,
                )
            )

        to_run.append(
            ExtractEFeatures(emodel=self.emodel, ttype=self.ttype, iteration_tag=self.iteration_tag)
        )

        return to_run

    def run(self):
        """ """
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        for seed in range(self.seed, self.seed + batch_size):
            # can have unfulfilled dependecies if slurm has send signal near time limit.
            if OptimisationTarget(
                emodel=self.emodel,
                ttype=self.ttype,
                iteration_tag=self.iteration_tag,
                seed=seed,
            ).exists():
                store_best_model(self.access_point, seed)

    def output(self):
        """ """
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        targets = []
        for seed in range(self.seed, self.seed + batch_size):
            targets.append(
                BestModelTarget(
                    emodel=self.emodel,
                    ttype=self.ttype,
                    iteration_tag=self.iteration_tag,
                    seed=seed,
                )
            )
        return targets


class ValidationTarget(WorkflowTarget):
    """Check if validation has been performed on the model.

    Return True if Validation task has already been performed on the model,
        even if the model is not validated.
    """

    def __init__(self, emodel, ttype, iteration_tag, seed):
        """Constructor.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            ttype (str): name of the ttype.
            seed (int): seed used in the optimisation.
        """
        super().__init__(emodel=emodel, ttype=ttype, iteration_tag=iteration_tag)

        self.seed = seed

    def exists(self):
        """Check if the model is completed for all given seeds."""
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        checked_for_all_seeds = [
            self.access_point.is_checked_by_validation(seed=seed)
            for seed in range(self.seed, self.seed + batch_size)
        ]
        return all(checked_for_all_seeds)


class Validation(WorkflowTaskRequiringMechanisms, IPyParallelTask):
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
                iteration_tag=self.iteration_tag,
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
                    iteration_tag=self.iteration_tag,
                    species=self.species,
                    brain_region=self.brain_region,
                )
            )

        if plot_optimisation:
            to_run.append(
                PlotModels(
                    emodel=self.emodel,
                    ttype=self.ttype,
                    iteration_tag=self.iteration_tag,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=self.seed,
                )
            )

        to_run.append(
            ExtractEFeatures(emodel=self.emodel, ttype=self.ttype, iteration_tag=self.iteration_tag)
        )

        return to_run

    def run(self):
        """Prepare self.args, then call bbp-workflow's IPyParallelTask's run()."""
        attrs = [
            "backend",
            "species",
            "emodel",
            "brain_region",
            "ttype",
            "iteration_tag",
        ]
        self.prepare_args_for_remote_script(attrs)

        super().run()

    def remote_script(self):
        """Catch arguments from parsing, and run validation."""
        # -- imports -- #
        import argparse
        import json

        from bluepyemodel import access_point
        from bluepyemodel.tools.multiprocessing import get_mapper
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
        parser.add_argument("--brain_region", default=None, type=str)
        parser.add_argument("--ttype", default=None, type=str)
        parser.add_argument("--iteration_tag", default=None, type=str)
        args = parser.parse_args()

        # -- run validation -- #
        mapper = get_mapper(args.backend)
        access_pt = access_point.get_access_point(
            access_point=args.api_from_config,
            emodel=args.emodel,
            ttype=args.ttype,
            iteration_tag=args.iteration_tag,
            **args.api_args_from_config,
        )

        validate(access_pt, mapper)

    def output(self):
        """ """
        return ValidationTarget(
            emodel=self.emodel,
            ttype=self.ttype,
            iteration_tag=self.iteration_tag,
            seed=self.seed,
        )


class EModelCreationTarget(WorkflowTarget):
    """Check if the the model is validated for any seed."""

    def __init__(self, emodel, ttype, iteration_tag):
        """Constructor.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            ttype (str): name of the ttype.
        """
        super().__init__(emodel=emodel, ttype=ttype, iteration_tag=iteration_tag)

    def exists(self):
        """Check if the model is completed."""

        return self.access_point.is_validated()


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
                    iteration_tag=self.iteration_tag,
                    species=self.species,
                    brain_region=self.brain_region,
                    seed=seed,
                )
            )
            seed += batch_size

        assert self.output().exists()

    def output(self):
        """ """
        return EModelCreationTarget(
            emodel=self.emodel, ttype=self.ttype, iteration_tag=self.iteration_tag
        )


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
                iteration_tag=self.iteration_tag,
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
                    iteration_tag=self.iteration_tag,
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
                    ttype=self.ttype,
                    iteration_tag=self.iteration_tag,
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
            iteration_tag=self.iteration_tag,
            species=self.species,
            brain_region=self.brain_region,
            seed=self.seed,
        )

    def run(self):
        """ """

        checkpoint_path = get_checkpoint_path(
            self.emodel,
            seed=self.seed,
            iteration_tag=self.iteration_tag,
            ttype=_reformat_ttype(self.ttype),
        )

        optimization(
            checkpoint_path=checkpoint_path,
            figures_dir=Path("./figures") / self.emodel / "optimisation",
        )

    def output(self):
        """ """

        chkpt_name = get_checkpoint_path(
            self.emodel,
            self.seed,
            iteration_tag=self.iteration_tag,
            ttype=_reformat_ttype(self.ttype),
        )

        fname = f"{Path(chkpt_name).stem}.pdf"
        return luigi.LocalTarget(Path("./figures") / self.emodel / "optimisation" / fname)


class PlotModels(WorkflowTaskRequiringMechanisms):
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
            ExtractEFeatures(
                emodel=self.emodel, ttype=self.ttype, iteration_tag=self.iteration_tag
            ),
            StoreBestModels(
                emodel=self.emodel,
                ttype=self.ttype,
                iteration_tag=self.iteration_tag,
                species=self.species,
                brain_region=self.brain_region,
                seed=self.seed,
            ),
            CompileMechanisms(
                emodel=self.emodel,
                ttype=self.ttype,
                iteration_tag=self.iteration_tag,
                species=self.species,
                brain_region=self.brain_region,
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
            fname = run_metadata_as_string(
                self.emodel,
                seed="",
                ttype=_reformat_ttype(self.ttype),
                iteration_tag=self.iteration_tag,
            )
            fname += "__parameters_distribution.pdf"
            fpath = Path("./figures") / self.emodel / "distributions" / "all" / fname
            outputs.append(luigi.LocalTarget(fpath))

            # scores
            for seed in range(self.seed, self.seed + batch_size):
                fname = run_metadata_as_string(
                    self.emodel,
                    seed,
                    ttype=_reformat_ttype(self.ttype),
                    iteration_tag=self.iteration_tag,
                )
                fname += "__scores.pdf"
                fpath = Path("./figures") / self.emodel / "scores" / "all" / fname
                outputs.append(luigi.LocalTarget(fpath))

            # traces
            for seed in range(self.seed, self.seed + batch_size):
                fname = run_metadata_as_string(
                    self.emodel,
                    seed,
                    ttype=_reformat_ttype(self.ttype),
                    iteration_tag=self.iteration_tag,
                )
                fname += "__traces.pdf"
                fpath = Path("./figures") / self.emodel / "traces" / "all" / fname
                outputs.append(luigi.LocalTarget(fpath))

        return outputs


class PlotValidatedDistributions(WorkflowTaskRequiringMechanisms):
    """Luigi wrapper for plotting the optimisation outputs.

    Parameters:
        species (str): name of the species.
        brain_region (str): name of the brain region.
    """

    species = luigi.Parameter(default="")
    brain_region = luigi.Parameter(default="")

    def requires(self):
        """ """

        return [
            EModelCreation(
                emodel=self.emodel,
                ttype=self.ttype,
                iteration_tag=self.iteration_tag,
                species=self.species,
                brain_region=self.brain_region,
            ),
            CompileMechanisms(
                emodel=self.emodel,
                ttype=self.ttype,
                iteration_tag=self.iteration_tag,
                species=self.species,
                brain_region=self.brain_region,
            ),
        ]

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
