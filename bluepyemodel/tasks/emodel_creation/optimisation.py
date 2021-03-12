"""Luigi tasks for emodel optimisation."""
import multiprocessing
from pathlib import Path

import luigi
from luigi_tools.task import ParamRef
from luigi_tools.task import copy_params

from bluepyemodel.emodel_pipeline.emodel_pipeline import extract_save_features_protocols
from bluepyemodel.optimisation import copy_and_compile_mechanisms
from bluepyemodel.optimisation import setup_and_run_optimisation
from bluepyemodel.optimisation import store_best_model
from bluepyemodel.tasks.emodel_creation.config import OptimizeConfig
from bluepyemodel.tasks.luigi_tools import BoolParameterCustom
from bluepyemodel.tasks.luigi_tools import WorkflowTarget
from bluepyemodel.tasks.luigi_tools import WorkflowTask
from bluepyemodel.tasks.luigi_tools import WorkflowWrapperTask
from bluepyemodel.validation.validation import validate


class EfeaturesProtocolsTarget(WorkflowTarget):
    """Target to check if efeatures and protocols are present in the database."""

    def __init__(self, emodel, species=None):
        """Constructor.

        Args:
            emodel (str): name of the emodel. Has to match the name of the emodel
                under which the configuration data are stored.
            species (str): name of the species.
        """
        super().__init__()

        self.emodel = emodel
        self.species = species

    def exists(self):
        """Check if the features and protocols have been created."""
        return self.emodel_db.has_protocols_and_features(self.emodel, species=self.species)


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

    emodel = luigi.Parameter()
    species = luigi.Parameter(default=None)

    threshold_nvalue_save = luigi.IntParameter(default=1)
    name_Rin_protocol = luigi.Parameter(default=None)
    name_rmp_protocol = luigi.Parameter(default=None)
    validation_protocols = luigi.DictParameter(default=None)
    plot = BoolParameterCustom(default=False)

    def run(self):
        """"""
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
        """"""
        return EfeaturesProtocolsTarget(self.emodel, species=self.species)


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

    emodel = luigi.Parameter()
    species = luigi.Parameter(default=None)
    mechanisms_dir = luigi.Parameter(default="mechanisms")
    copy_mechanisms = BoolParameterCustom(default=False)

    def run(self):
        """"""
        copy_and_compile_mechanisms(
            self.emodel_db,
            self.emodel,
            self.species,
            self.copy_mechanisms,
            self.mechanisms_dir,
            githash="",
        )

    def output(self):
        """"""
        return luigi.LocalTarget(Path("x86_64") / "special")


class OptimisationTarget(WorkflowTarget):
    """Target to check if an optimisation is present in the database."""

    def __init__(
        self,
        emodel,
        species=None,
        seed=1,
        checkpoint_dir=None,
    ):
        """Constructor.

        Args:
           emodel (str): name of the emodel. Has to match the name of the emodel
               under which the configuration data are stored.
           species (str): name of the species.
           seed (int): seed used in the optimisation.
           checkpoint_dir (str): path to the repo where files used as a checkpoint by BluePyOpt are.
        """
        super().__init__()

        self.emodel = emodel
        self.species = species
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed

    def exists(self):
        """Check if the model is completed."""
        return self.emodel_db.optimisation_state(
            self.emodel,
            self.checkpoint_dir,
            species=self.species,
            seed=self.seed,
            githash="",
        )


@copy_params(
    mechanisms_dir=ParamRef(OptimizeConfig),
    morphology_modifiers=ParamRef(OptimizeConfig),
    max_ngen=ParamRef(OptimizeConfig),
    stochasticity=ParamRef(OptimizeConfig),
    copy_mechanisms=ParamRef(OptimizeConfig),
    compile_mechanisms=ParamRef(OptimizeConfig),
    opt_params=ParamRef(OptimizeConfig),
    optimizer=ParamRef(OptimizeConfig),
    checkpoint_dir=ParamRef(OptimizeConfig),
    continue_opt=ParamRef(OptimizeConfig),
    timeout=ParamRef(OptimizeConfig),
)
class Optimize(WorkflowTask):
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
    emodel = luigi.Parameter(default=None)
    species = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=42)
    graceful_killer = multiprocessing.Event()

    def requires(self):
        """"""
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
        """"""
        # terminate = GracefulKiller()
        if not self.graceful_killer.is_set():
            mapper = self.get_mapper()
            setup_and_run_optimisation(
                self.emodel_db,
                self.emodel,
                self.seed,
                species=self.species,
                morphology_modifiers=self.morphology_modifiers,
                stochasticity=self.stochasticity,
                include_validation_protocols=False,
                optimisation_rules=None,
                timeout=self.timeout,
                mapper=mapper,
                opt_params=self.opt_params,  # these should be real parameters from luigi.cfg
                optimizer=self.optimizer,
                max_ngen=self.max_ngen,
                checkpoint_dir=self.checkpoint_dir,
                continue_opt=self.continue_opt,
                githash="",
                terminator=self.graceful_killer,
            )

    def output(self):
        """"""
        return OptimisationTarget(
            emodel=self.emodel,
            species=self.species,
            checkpoint_dir=self.checkpoint_dir,
            seed=self.seed,
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
        super().__init__()

        self.emodel = emodel
        self.seed = seed

    def exists(self):
        """Check if the best model is stored."""
        return self.emodel_db.has_best_model(emodel=self.emodel, seed=self.seed, githash="")


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
    """

    emodel = luigi.Parameter()
    species = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=42)
    batch_size = luigi.IntParameter(default=1)

    def requires(self):
        """"""
        to_run = []

        for seed in range(self.seed, self.seed + self.batch_size):
            to_run.append(Optimize(emodel=self.emodel, species=self.species, seed=seed))
        return to_run

    def run(self):
        """"""
        for seed in range(self.seed, self.seed + self.batch_size):
            # can have unfulfilled dependecies if slurm has send signal near time limit.
            if BestModelTarget(emodel=self.emodel, seed=seed).exists():
                store_best_model(
                    self.emodel_db,
                    self.emodel,
                    self.species,
                    seed,
                    stochasticity=self.stochasticity,
                    include_validation_protocols=False,
                    optimisation_rules=None,
                    optimizer=self.optimizer,
                    checkpoint_dir=self.checkpoint_dir,
                    githash="",
                )

    def output(self):
        """"""
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
        super().__init__()

        self.emodel = emodel
        self.seed = seed
        self.batch_size = batch_size

    def exists(self):
        """Check if the model is completed for all given seeds."""
        checked_for_all_seeds = [
            self.emodel_db.is_checked_by_validation(emodel=self.emodel, seed=seed, githash="")
            for seed in range(self.seed, self.seed + self.batch_size)
        ]
        return all(checked_for_all_seeds)


@copy_params(
    stochasticity=ParamRef(OptimizeConfig),
    copy_mechanisms=ParamRef(OptimizeConfig),
    compile_mechanisms=ParamRef(OptimizeConfig),
    mechanisms_dir=ParamRef(OptimizeConfig),
    morphology_modifiers=ParamRef(OptimizeConfig),
)
class Validation(WorkflowTask):
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
        graceful_killer (multiprocessing.Event): event triggered when USR1 signal is received.
            Has to use multiprocessing event for communicating between processes
            when there is more than 1 luigi worker. Skip task if set.
    """

    emodel = luigi.Parameter()
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
    graceful_killer = multiprocessing.Event()

    def requires(self):
        """"""
        to_run = [StoreBestModels(emodel=self.emodel, species=self.species, seed=self.seed)]
        if self.compile_mechanisms:
            to_run.append(
                CompileMechanisms(
                    emodel=self.emodel,
                    species=self.species,
                    mechanisms_dir=self.mechanisms_dir,
                    copy_mechanisms=self.copy_mechanisms,
                )
            )
        return to_run

    def run(self):
        """"""
        if not self.graceful_killer.is_set():
            mapper = self.get_mapper()
            validate(
                self.emodel_db,
                self.emodel,
                self.species,
                mapper,
                validation_function=self.validation_function,
                stochasticity=self.stochasticity,
                morphology_modifiers=self.morphology_modifiers,
                additional_protocols=self.additional_protocols,
                threshold=self.threshold,
                validation_protocols_only=self.validation_protocols_only,
            )

        assert self.output().exists()

    def output(self):
        """"""
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
        super().__init__()

        self.emodel = emodel
        self.n_models_to_pass_validation = n_models_to_pass_validation

    def exists(self):
        """Check if the model is completed."""
        return self.emodel_db.is_validated(
            emodel=self.emodel,
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

    emodel = luigi.Parameter()
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
        """"""
        return EModelCreationTarget(
            emodel=self.emodel,
            n_models_to_pass_validation=self.n_models_to_pass_validation,
        )


class OptimizeWrapper(WorkflowWrapperTask):
    """Luigi wrapper for launching multiple seeds to optimize.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        seed (int): seed used in the optimisation.
        batch_size (int): number of seeds to optimize at the same time before each validation.
    """

    emodel = luigi.Parameter()
    species = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=42)
    batch_size = luigi.IntParameter(default=10)

    def requires(self):
        """"""
        to_run = []
        for seed in range(self.seed, self.seed + self.batch_size):
            to_run.append(Optimize(emodel=self.emodel, species=self.species, seed=seed))
        return to_run
