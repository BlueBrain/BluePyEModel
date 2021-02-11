"""Luigi tasks for emodel optimisation."""
import logging

import luigi

from bluepyemodel.emodel_pipeline.emodel_pipeline import extract_save_features_protocols
from bluepyemodel.optimisation import setup_and_run_optimisation
from bluepyemodel.tasks.luigi_tools import BoolParameterCustom
from bluepyemodel.tasks.luigi_tools import WorkflowTarget
from bluepyemodel.tasks.luigi_tools import WorkflowTask

logger = logging.getLogger(__name__)


class EfeaturesProtocolsTarget(WorkflowTarget):
    """Target to check if efeatures and protocols are present in the postgreSQL database."""

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
        config_dict (dict): BluePyEfe configuration dictionnary. Only required if
            db_api is 'singlecell'
        threshold_nvalue_save (int): lower bounds of the number of values required
            to save an efeature.
        name_Rin_protocol (str): name of the protocol that should be used to compute
            the input resistance. Only used when db_api is 'singlecell'
        name_rmp_protocol (str): name of the protocol that should be used to compute
            the resting membrane potential. Only used when db_api is 'singlecell'.
        validation_protocols (dict): Of the form {"ecodename": [targets]}. Only used
            when db_api is 'singlecell'.
    """

    emodel = luigi.Parameter()
    species = luigi.Parameter(default=None)

    threshold_nvalue_save = luigi.IntParameter(default=1)
    config_dict = luigi.Parameter(default=None)
    name_Rin_protocol = luigi.Parameter(default=None)
    name_rmp_protocol = luigi.Parameter(default=None)
    validation_protocols = luigi.DictParameter(default=None)

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
        )

    def output(self):
        """"""
        return EfeaturesProtocolsTarget(self.emodel, species=self.species)


class OptimisationTarget(WorkflowTarget):
    """Target to check if an optimisation is present in the postgreSQL database."""

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
            self.emodel, self.checkpoint_dir, species=self.species, seed=self.seed
        )


class Optimize(WorkflowTask):
    """Luigi wrapper for emodel_pipeline.emodel_creation.optimize

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        species (str): name of the species.
        seed (int): seed used in the optimisation.
        mechanisms_dir (str): path of the directory in which the mechanisms
            will be copied and/or compiled. It has to be a subdirectory of
            working_dir.
        morphology_modifiers (list): list of python functions that will be
            applied to all the morphologies.
        max_ngen (int): maximum number of generations of the evolutionary process.
        stochasticity (bool): should channels behave stochastically if they can.
        copy_mechanisms (bool): should the mod files be copied in the local
            mechanisms_dir directory.
        compile_mechanisms (bool): should the mod files be compiled.
        opt_params (dict): optimisation parameters. Keys have to match the
            optimizer's call.
        optimizer (str): algorithm used for optimization, can be "IBEA", "SO-CMA",
            "MO-CMA".
        checkpoint_dir (str): path to the repo where files used as a checkpoint by BluePyOpt are.
        continue_opt (bool): should the optimization restart from a previously
            created checkpoint file.
        timeout (float): duration (in second) after which the evaluation of a
            protocol will be interrupted.
    """

    emodel = luigi.Parameter()
    species = luigi.Parameter(default=None)
    seed = luigi.IntParameter(default=42)

    mechanisms_dir = luigi.Parameter(default="mechanisms")
    morphology_modifiers = luigi.ListParameter(default=None)
    max_ngen = luigi.IntParameter(default=1000)
    stochasticity = BoolParameterCustom(default=False)
    copy_mechanisms = BoolParameterCustom(default=False)
    compile_mechanisms = BoolParameterCustom(default=False)
    opt_params = luigi.DictParameter(default=None)
    optimizer = luigi.Parameter(default="MO-CMA")
    checkpoint_dir = luigi.Parameter("./checkpoints/")
    continue_opt = BoolParameterCustom(default=False)
    timeout = luigi.IntParameter(default=600)

    def requires(self):
        """"""
        return ExtractEFeatures(emodel=self.emodel, species=self.species)

    def run(self):
        """"""
        mapper = self.get_mapper()
        setup_and_run_optimisation(
            self.emodel_db,
            self.emodel,
            self.seed,
            species=self.species,
            mechanisms_dir=self.mechanisms_dir,
            morphology_modifiers=self.morphology_modifiers,
            stochasticity=self.stochasticity,
            copy_mechanisms=self.copy_mechanisms,
            compile_mechanisms=self.compile_mechanisms,
            include_validation_protocols=False,
            optimisation_rules=None,
            timeout=self.timeout,
            mapper=mapper,
            opt_params=self.opt_params,  # these should be real parameters from luigi.cfg
            optimizer=self.optimizer,
            max_ngen=self.max_ngen,
            checkpoint_dir=self.checkpoint_dir,
            continue_opt=self.continue_opt,
        )

    def output(self):
        """"""
        return OptimisationTarget(
            emodel=self.emodel,
            species=self.species,
            checkpoint_dir=self.checkpoint_dir,
            seed=self.seed,
        )
