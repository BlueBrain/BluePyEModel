"""Luigi tasks for emodel optimisation."""
import glob
import multiprocessing
from pathlib import Path

import luigi

from bluepyemodel.access_point.forge_access_point import AccessPointException
from bluepyemodel.efeatures_extraction.efeatures_extraction import extract_save_features_protocols
from bluepyemodel.efeatures_extraction.targets_configurator import TargetsConfigurator
from bluepyemodel.emodel_pipeline.plotting import optimisation
from bluepyemodel.emodel_pipeline.plotting import plot_models
from bluepyemodel.optimisation import get_checkpoint_path
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.tasks.luigi_tools import IPyParallelTask
from bluepyemodel.tasks.luigi_tools import WorkflowTarget
from bluepyemodel.tasks.luigi_tools import WorkflowTask
from bluepyemodel.tasks.luigi_tools import WorkflowTaskRequiringMechanisms
from bluepyemodel.tasks.luigi_tools import WorkflowWrapperTask
from bluepyemodel.tools.mechanisms import compile_mechs_in_emodel_dir
from bluepyemodel.tools.utils import get_legacy_checkpoint_path

# pylint: disable=W0235,W0621,W0404,W0611,W0703


def _reformat_ttype(ttype):
    """ """

    if isinstance(ttype, str):
        return ttype.replace("__", " ")

    return None


class CreateTargetsConfigurationTarget(WorkflowTarget):
    """Luigi target to check if BPEM targets configuraiton for extraction exists."""

    def __init__(self, emodel, etype, ttype, mtype, species, brain_region, iteration_tag):
        """Constructor."""
        super().__init__(
            emodel=emodel,
            etype=etype,
            ttype=ttype,
            mtype=mtype,
            species=species,
            brain_region=brain_region,
            iteration_tag=iteration_tag,
        )

    def exists(self):
        """Check if the targets configuration has been created."""
        return self.access_point.has_targets_configuration()


class CreateTargetsConfiguration(WorkflowTask):
    """Luigi wrapper for create_targets in emodel_pipeline.EModel_pipeline."""

    @WorkflowTask.check_mettypes
    def run(self):
        """ """

        configurator = TargetsConfigurator(access_point=self.access_point)
        configurator.create_and_save_configuration_from_access_point()

    def output(self):
        """ """
        return CreateTargetsConfigurationTarget(
            emodel=self.emodel,
            etype=self.etype,
            ttype=self.ttype,
            mtype=self.mtype,
            species=self.species,
            brain_region=self.brain_region,
            iteration_tag=self.iteration_tag,
        )


class EfeaturesProtocolsTarget(WorkflowTarget):
    """Target to check if efeatures and protocols are present in the database."""

    def __init__(self, emodel, etype, ttype, mtype, species, brain_region, iteration_tag):
        """Constructor."""
        super().__init__(
            emodel=emodel,
            etype=etype,
            ttype=ttype,
            mtype=mtype,
            species=species,
            brain_region=brain_region,
            iteration_tag=iteration_tag,
        )

    def exists(self):
        """Check if the features and protocols have been created."""

        return self.access_point.has_fitness_calculator_configuration()


class ExtractEFeatures(WorkflowTask):
    """Luigi wrapper for extract_efeatures in emodel_pipeline.EModel_pipeline."""

    def requires(self):
        """ """
        return CreateTargetsConfiguration(
            emodel=self.emodel,
            etype=self.etype,
            ttype=self.ttype,
            mtype=self.mtype,
            species=self.species,
            brain_region=self.brain_region,
            iteration_tag=self.iteration_tag,
        )

    @WorkflowTask.check_mettypes
    def run(self):
        """ """

        mapper = self.get_mapper()

        _ = extract_save_features_protocols(access_point=self.access_point, mapper=mapper)

    def output(self):
        """ """
        return EfeaturesProtocolsTarget(
            emodel=self.emodel,
            etype=self.etype,
            ttype=self.ttype,
            mtype=self.mtype,
            species=self.species,
            brain_region=self.brain_region,
            iteration_tag=self.iteration_tag,
        )


class CompileMechanisms(WorkflowTaskRequiringMechanisms):
    """Luigi wrapper for mechanisms.compile_mechs_in_emodel_dir"""

    def run(self):
        """ """

        mechanisms_directory = self.access_point.get_mechanisms_directory()
        compile_mechs_in_emodel_dir(mechanisms_directory)

    def output(self):
        """ """

        targets = []
        mechanisms_directory = self.access_point.get_mechanisms_directory()

        for filepath in glob.glob(f"{str(mechanisms_directory)}/*.mod"):
            compile_path = mechanisms_directory.parents[0] / "x86_64" / f"{Path(filepath).stem}.c"
            targets.append(luigi.LocalTarget(compile_path))

        return targets


class OptimisationTarget(WorkflowTarget):
    """Target to check if an optimisation is present in the database."""

    def __init__(
        self,
        emodel,
        etype,
        ttype,
        mtype,
        species,
        brain_region,
        iteration_tag,
        seed=1,
        continue_opt=False,
    ):
        """Constructor.

        Args:
           seed (int): seed used in the optimisation.
           continue_opt (bool): whether to continue optimisation or not
                when the optimisation is not complete.
        """
        super().__init__(
            emodel=emodel,
            etype=etype,
            ttype=ttype,
            mtype=mtype,
            species=species,
            brain_region=brain_region,
            iteration_tag=iteration_tag,
        )

        self.seed = seed
        self.continue_opt = continue_opt

    def exists(self):
        """Check if the model is completed."""
        return self.access_point.optimisation_state(seed=self.seed, continue_opt=self.continue_opt)


class Optimise(WorkflowTaskRequiringMechanisms, IPyParallelTask):
    """Luigi wrapper for emodel_pipeline.emodel_creation.optimise

    Parameters:
        seed (int): seed used in the optimisation.
        continue_unfinished_optimisation (bool): whether to continue optimisation or not
                when the optimisation is not complete.
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker.
            When set, will gracefully exit main loop in Optimise (in deap algorithm).
    """

    # if default not set, crashes when parameters are read by luigi_tools.copy_params
    seed = luigi.IntParameter(default=42)
    continue_unfinished_optimisation = luigi.BoolParameter(default=False)
    graceful_killer = multiprocessing.Event()

    def requires(self):
        """ """
        compile_mechanisms = self.access_point.pipeline_settings.compile_mechanisms

        targets = [
            ExtractEFeatures(
                emodel=self.emodel,
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
                iteration_tag=self.iteration_tag,
            )
        ]

        if compile_mechanisms:
            targets.append(
                CompileMechanisms(
                    emodel=self.emodel,
                    etype=self.etype,
                    ttype=self.ttype,
                    mtype=self.mtype,
                    species=self.species,
                    brain_region=self.brain_region,
                    iteration_tag=self.iteration_tag,
                )
            )

        return targets

    @WorkflowTask.check_mettypes
    def run(self):
        """Prepare self.args, then call bbp-workflow's IPyParallelTask's run()."""
        attrs = [
            "backend",
            "emodel",
            "seed",
            "etype",
            "ttype",
            "mtype",
            "species",
            "brain_region",
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
        parser.add_argument("--etype", default=None, type=str)
        parser.add_argument("--ttype", default=None, type=str)
        parser.add_argument("--mtype", default=None, type=str)
        parser.add_argument("--species", default=None, type=str)
        parser.add_argument("--brain_region", default=None, type=str)
        parser.add_argument("--iteration_tag", default=None, type=str)
        parser.add_argument("--seed", default=1, type=int)
        parser.add_argument("--ipyparallel_profile", default=None, type=str)
        args = parser.parse_args()

        # -- run optimisation -- #
        mapper = get_mapper(args.backend, ipyparallel_profile=args.ipyparallel_profile)
        access_pt = access_point.get_access_point(
            access_point=args.api_from_config,
            emodel=args.emodel,
            etype=args.etype,
            ttype=args.ttype,
            mtype=args.mtype,
            species=args.species,
            brain_region=args.brain_region,
            iteration_tag=args.iteration_tag,
            **args.api_args_from_config,
        )
        setup_and_run_optimisation(access_pt, args.seed, mapper=mapper)

    def output(self):
        """ """
        return OptimisationTarget(
            emodel=self.emodel,
            etype=self.etype,
            ttype=self.ttype,
            mtype=self.mtype,
            species=self.species,
            brain_region=self.brain_region,
            iteration_tag=self.iteration_tag,
            seed=self.seed,
            continue_opt=self.continue_unfinished_optimisation,
        )


class BestModelTarget(WorkflowTarget):
    """Check if the best model from optimisation is present in the database."""

    def __init__(self, emodel, etype, ttype, mtype, species, brain_region, iteration_tag, seed=1):
        """Constructor.

        Args:
            seed (int): seed used in the optimisation.
        """
        super().__init__(
            emodel=emodel,
            etype=etype,
            ttype=ttype,
            mtype=mtype,
            species=species,
            brain_region=brain_region,
            iteration_tag=iteration_tag,
        )

        self.seed = seed

    def exists(self):
        """Check if the best model is stored."""
        return self.access_point.has_best_model(seed=self.seed)


class StoreBestModels(WorkflowTaskRequiringMechanisms):
    """Luigi wrapper for store_best_model.

    Parameters:
        seed (int): seed used in the optimisation.
    """

    seed = luigi.IntParameter(default=1)

    def requires(self):
        """ """
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size

        to_run = []

        for seed in range(self.seed, self.seed + batch_size):
            to_run.append(
                Optimise(
                    emodel=self.emodel,
                    etype=self.etype,
                    ttype=self.ttype,
                    mtype=self.mtype,
                    species=self.species,
                    brain_region=self.brain_region,
                    iteration_tag=self.iteration_tag,
                    seed=seed,
                )
            )

            to_run.append(
                PlotOptimisation(
                    emodel=self.emodel,
                    etype=self.etype,
                    ttype=self.ttype,
                    mtype=self.mtype,
                    species=self.species,
                    brain_region=self.brain_region,
                    iteration_tag=self.iteration_tag,
                    seed=seed,
                )
            )

        to_run.append(
            ExtractEFeatures(
                emodel=self.emodel,
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
                iteration_tag=self.iteration_tag,
            )
        )

        return to_run

    @WorkflowTask.check_mettypes
    def run(self):
        """ """
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        for seed in range(self.seed, self.seed + batch_size):
            # can have unfulfilled dependecies if slurm has send signal near time limit.
            if OptimisationTarget(
                emodel=self.emodel,
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
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
                    etype=self.etype,
                    ttype=self.ttype,
                    mtype=self.mtype,
                    species=self.species,
                    brain_region=self.brain_region,
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

    def __init__(self, emodel, etype, ttype, mtype, species, brain_region, iteration_tag, seed):
        """Constructor.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            ttype (str): name of the ttype.
            seed (int): seed used in the optimisation.
        """
        super().__init__(
            emodel=emodel,
            etype=etype,
            ttype=ttype,
            mtype=mtype,
            species=species,
            brain_region=brain_region,
            iteration_tag=iteration_tag,
        )

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
        seed (int): seed used in the optimisation.
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker. Skip task if set.
    """

    seed = luigi.IntParameter(default=1)
    graceful_killer = multiprocessing.Event()

    def requires(self):
        """ """

        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation
        compile_mechanisms = self.access_point.pipeline_settings.compile_mechanisms

        to_run = [
            StoreBestModels(
                emodel=self.emodel,
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
                iteration_tag=self.iteration_tag,
                seed=self.seed,
            )
        ]
        if compile_mechanisms:
            to_run.append(
                CompileMechanisms(
                    emodel=self.emodel,
                    etype=self.etype,
                    ttype=self.ttype,
                    mtype=self.mtype,
                    species=self.species,
                    brain_region=self.brain_region,
                    iteration_tag=self.iteration_tag,
                )
            )

        if plot_optimisation:
            to_run.append(
                PlotModels(
                    emodel=self.emodel,
                    etype=self.etype,
                    ttype=self.ttype,
                    mtype=self.mtype,
                    species=self.species,
                    brain_region=self.brain_region,
                    iteration_tag=self.iteration_tag,
                    seed=self.seed,
                )
            )

        to_run.append(
            ExtractEFeatures(
                emodel=self.emodel,
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
                iteration_tag=self.iteration_tag,
            )
        )

        return to_run

    @WorkflowTask.check_mettypes
    def run(self):
        """Prepare self.args, then call bbp-workflow's IPyParallelTask's run()."""
        attrs = [
            "backend",
            "emodel",
            "etype",
            "ttype",
            "mtype",
            "species",
            "brain_region",
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
        parser.add_argument("--etype", default=None, type=str)
        parser.add_argument("--ttype", default=None, type=str)
        parser.add_argument("--mtype", default=None, type=str)
        parser.add_argument("--species", default=None, type=str)
        parser.add_argument("--brain_region", default=None, type=str)
        parser.add_argument("--iteration_tag", default=None, type=str)
        parser.add_argument("--ipyparallel_profile", default=None, type=str)

        args = parser.parse_args()

        # -- run validation -- #
        mapper = get_mapper(args.backend, ipyparallel_profile=args.ipyparallel_profile)
        access_pt = access_point.get_access_point(
            access_point=args.api_from_config,
            emodel=args.emodel,
            etype=args.etype,
            ttype=args.ttype,
            mtype=args.mtype,
            species=args.species,
            brain_region=args.brain_region,
            iteration_tag=args.iteration_tag,
            **args.api_args_from_config,
        )

        validate(access_pt, mapper)

    def output(self):
        """ """
        return ValidationTarget(
            emodel=self.emodel,
            etype=self.etype,
            ttype=self.ttype,
            mtype=self.mtype,
            species=self.species,
            brain_region=self.brain_region,
            iteration_tag=self.iteration_tag,
            seed=self.seed,
        )


class EModelCreationTarget(WorkflowTarget):
    """Check if the the model is validated for any seed."""

    def __init__(self, emodel, etype, ttype, mtype, species, brain_region, iteration_tag):
        """Constructor.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            ttype (str): name of the ttype.
        """
        super().__init__(
            emodel=emodel,
            etype=etype,
            ttype=ttype,
            mtype=mtype,
            species=species,
            brain_region=brain_region,
            iteration_tag=iteration_tag,
        )

    def update_emodel_workflow(self, state="done"):
        """Update emodel workflow state on nexus"""
        # get / create
        emodel_workflow = self.access_point.get_emodel_workflow()
        if emodel_workflow is None:
            emodel_workflow = self.access_point.create_emodel_workflow()

        # store / update
        # this function might be called several times by luigi, so
        # only update the resource when necessary
        if emodel_workflow.state != state:
            emodel_workflow.state = state
            self.access_point.store_or_update_emodel_workflow(emodel_workflow)

    def exists(self):
        """Check if the model is completed."""
        exist = self.access_point.is_validated()
        if exist:
            # only for nexus access_point
            self.update_emodel_workflow(state="done")

        return exist


class EModelCreation(WorkflowTask):
    """Main Workflow Task. Creates an emodel.

    Parameters:
        emodel (str): name of the emodel.
        ttype (str): name of the t-type.
        species (str): name of the species.
        brain_region (str): name of the brain region.
        seed (int): seed used in the optimisation.
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker. Exit loop if set.
    """

    seed = luigi.IntParameter(default=1)
    graceful_killer = multiprocessing.Event()

    def check_and_update_emodel_workflow(self, state="running"):
        """Get or create emodel workflow, check its configuration, and update its state on nexus"""
        # get / create
        emodel_workflow = self.access_point.get_emodel_workflow()
        if emodel_workflow is None:
            emodel_workflow = self.access_point.create_emodel_workflow()

        # check
        has_configuration = self.access_point.check_emodel_workflow_configurations(emodel_workflow)
        if not has_configuration:
            raise AccessPointException(
                "There are configuration files missing on nexus for the workflow."
                "Please check that you have registered on nexus the following resources:\n"
                "ExtractionTargetsConfiguration\n"
                "EModelPipelineSettings\n"
                "EModelConfiguration"
            )

        # store / update
        emodel_workflow.state = state
        self.access_point.store_or_update_emodel_workflow(emodel_workflow)

    @WorkflowTask.check_mettypes
    def run(self):
        """Optimise e-models by batches of batch_size until one is validated."""

        seed = self.seed

        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        max_n_batch = self.access_point.pipeline_settings.max_n_batch

        # only for nexus access_point
        self.check_and_update_emodel_workflow(state="running")

        while not self.output().exists() and not self.graceful_killer.is_set():
            # limit the number of batch
            if seed > self.seed + max_n_batch * batch_size:
                break

            yield (
                Validation(
                    emodel=self.emodel,
                    etype=self.etype,
                    ttype=self.ttype,
                    mtype=self.mtype,
                    species=self.species,
                    brain_region=self.brain_region,
                    iteration_tag=self.iteration_tag,
                    seed=seed,
                )
            )
            seed += batch_size

        assert self.output().exists()

    def output(self):
        """ """
        return EModelCreationTarget(
            emodel=self.emodel,
            etype=self.etype,
            ttype=self.ttype,
            mtype=self.mtype,
            species=self.species,
            brain_region=self.brain_region,
            iteration_tag=self.iteration_tag,
        )


class SeveralEModelCreation(luigi.WrapperTask):
    """Wrapper around EModelCreation. Creates several emodels.

    Parameters:
        emodels (str): name of the emodels, separated by ",".
        etypes (str): None, etype or list of etypes of the emodels, separated by ",". Optional.
        ttypes (str): None, ttype or list of ttypes of the emodels, separated by ",". Optional.
        mtypes (str): None, mtype or list of mtypes of the emodels, separated by ",". Optional.
        species (str): name of the species. Optional.
        brain_region (str): name of the brain region. Optional.
        iteration_tag (str): iteration tag or githash to identify the run. Optional.
    """

    emodels = luigi.Parameter(default=[])
    etypes = luigi.Parameter(default=None)
    ttypes = luigi.Parameter(default=None)
    mtypes = luigi.Parameter(default=None)
    species = luigi.Parameter(default=None)
    brain_region = luigi.Parameter(default=None)
    iteration_tag = luigi.Parameter(default=None)

    @staticmethod
    def _parse_emodel_metadata(metadata, n_models):
        """Parse the string "etype1,etype2,..." into ["etype1", "etype2", ...]"""
        # pylint: disable=no-member, unsupported-membership-test
        if metadata is None or "," not in metadata:
            return [metadata] * n_models
        return metadata.split(",")

    def requires(self):
        # pylint: disable=no-member
        emodels = self.emodels.split(",")
        etypes = self._parse_emodel_metadata(self.etypes, len(emodels))
        ttypes = self._parse_emodel_metadata(self.ttypes, len(emodels))
        mtypes = self._parse_emodel_metadata(self.mtypes, len(emodels))

        to_run = []
        for emodel, etype, ttype, mtype in zip(emodels, etypes, ttypes, mtypes):
            to_run.append(
                EModelCreation(
                    emodel=emodel,
                    etype=etype,
                    ttype=ttype,
                    mtype=mtype,
                    species=self.species,
                    brain_region=self.brain_region,
                    iteration_tag=self.iteration_tag,
                )
            )

        return to_run


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
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
                iteration_tag=self.iteration_tag,
            )
        ]

        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation

        if plot_optimisation:
            to_run.append(
                PlotValidatedDistributions(
                    emodel=self.emodel,
                    etype=self.etype,
                    ttype=self.ttype,
                    mtype=self.mtype,
                    species=self.species,
                    brain_region=self.brain_region,
                    iteration_tag=self.iteration_tag,
                )
            )

        return to_run


class OptimiseWrapper(WorkflowWrapperTask):
    """Luigi wrapper for launching multiple seeds to optimise.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        ttype (str): name of the t-type.
        species (str): name of the species.
        brain_region (str): name of the brain region.
        seed (int): seed used in the optimisation.
        batch_size (int): number of seeds to optimise at the same time before each validation.
    """

    seed = luigi.IntParameter(default=42)

    def requires(self):
        """ """
        to_run = []

        batch_size = self.access_point.pipeline_settings.optimisation_batch_size

        for seed in range(self.seed, self.seed + batch_size):
            to_run.append(
                Optimise(
                    emodel=self.emodel,
                    etype=self.etype,
                    ttype=self.ttype,
                    mtype=self.mtype,
                    species=self.species,
                    brain_region=self.brain_region,
                    iteration_tag=self.iteration_tag,
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

    seed = luigi.IntParameter(default=42)

    def requires(self):
        """ """
        return Optimise(
            emodel=self.emodel,
            etype=self.etype,
            ttype=self.ttype,
            mtype=self.mtype,
            species=self.species,
            brain_region=self.brain_region,
            iteration_tag=self.iteration_tag,
            seed=self.seed,
        )

    def run(self):
        """ """

        checkpoint_path = get_checkpoint_path(self.access_point.emodel_metadata, seed=self.seed)
        if (
            not Path(checkpoint_path).is_file()
            and Path(get_legacy_checkpoint_path(checkpoint_path)).is_file()
        ):
            checkpoint_path = get_legacy_checkpoint_path(checkpoint_path)

        optimisation(
            optimiser=self.access_point.pipeline_settings.optimiser,
            emodel=self.emodel,
            iteration=self.iteration_tag,
            seed=self.seed,
            checkpoint_path=checkpoint_path,
            figures_dir=Path("./figures") / self.emodel / "optimisation",
        )

    def output(self):
        """ """
        checkpoint_path = get_checkpoint_path(self.access_point.emodel_metadata, seed=self.seed)

        fname = f"{Path(checkpoint_path).stem}.pdf"
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
                emodel=self.emodel,
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
                iteration_tag=self.iteration_tag,
            ),
            StoreBestModels(
                emodel=self.emodel,
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
                iteration_tag=self.iteration_tag,
                seed=self.seed,
            ),
            CompileMechanisms(
                emodel=self.emodel,
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
                iteration_tag=self.iteration_tag,
            ),
        ]

        return requires

    def run(self):
        """ """

        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation
        plot_currentscape = self.access_point.pipeline_settings.plot_currentscape
        batch_size = self.access_point.pipeline_settings.optimisation_batch_size

        mapper = self.get_mapper()
        plot_models(
            access_point=self.access_point,
            mapper=mapper,
            seeds=range(self.seed, self.seed + batch_size),
            figures_dir=Path("./figures") / self.emodel,
            plot_distributions=plot_optimisation,
            plot_traces=plot_optimisation,
            plot_scores=plot_optimisation,
            plot_currentscape=plot_currentscape,
        )

    def output(self):
        """ """

        batch_size = self.access_point.pipeline_settings.optimisation_batch_size
        plot_optimisation = self.access_point.pipeline_settings.plot_optimisation

        outputs = []
        if plot_optimisation:
            # distribution
            fname = self.access_point.emodel_metadata.as_string()
            fname += "__parameters_distribution.pdf"
            fpath = Path("./figures") / self.emodel / "distributions" / "all" / fname
            outputs.append(luigi.LocalTarget(fpath))

            # scores
            for seed in range(self.seed, self.seed + batch_size):
                fname = self.access_point.emodel_metadata.as_string(seed)
                fname += "__scores.pdf"
                fpath = Path("./figures") / self.emodel / "scores" / "all" / fname
                outputs.append(luigi.LocalTarget(fpath))

            # traces
            for seed in range(self.seed, self.seed + batch_size):
                fname = self.access_point.emodel_metadata.as_string(seed)
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

    def requires(self):
        """ """

        return [
            EModelCreation(
                emodel=self.emodel,
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
                iteration_tag=self.iteration_tag,
            ),
            CompileMechanisms(
                emodel=self.emodel,
                etype=self.etype,
                ttype=self.ttype,
                mtype=self.mtype,
                species=self.species,
                brain_region=self.brain_region,
                iteration_tag=self.iteration_tag,
            ),
        ]

    def run(self):
        """ """
        plot_models(
            access_point=self.access_point,
            mapper=self.get_mapper(),
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
