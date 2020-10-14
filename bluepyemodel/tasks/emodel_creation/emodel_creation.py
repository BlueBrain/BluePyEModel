import logging
import luigi

from bluepyemodel.emodel_pipeline.emodel_creation import (
    extract_efeatures,
    optimize,
)
from bluepyemodel.api.postgreSQL import PostgreSQL_API

logger = logging.getLogger(__name__)

# pylint: disable-all


class EfeaturesProtocolsTarget(luigi.Target):

    """
    Target to check if efeatures and protocols are present in the
    postgreSQL database.
    """

    def __init__(
        self,
        emodel,
        species,
    ):
        """
        Args:
            emodel (str): name of the emodel.
            species (str): name of the species (rat, human, mouse).
        """

        self.db = PostgreSQL_API()

        self.emodel = emodel
        self.species = species

    def exists(self):
        """
        Check if the efeatures and protocol for the emodel/species
        combo are present in the table.
        """

        targets = self.db.fetch(
            table="optimisation_targets",
            conditions={"emodel": self.emodel, "species": self.species},
        )

        tot_features = 0
        present_features = 0
        for target in targets.to_dict(orient="records"):

            # Check if protocol
            protocols = self.db.fetch(
                table="extraction_protocols",
                conditions={
                    "emodel": self.emodel,
                    "species": self.species,
                    "name": f"{target['ecode']}_{target['target']}",
                },
            )
            if protocols.empty:
                return False

            # Check if efeatures
            for feat in target["efeatures"]:
                tot_features += 1
                efeatures = self.db.fetch(
                    table="extraction_efeatures",
                    conditions={
                        "emodel": self.emodel,
                        "species": self.species,
                        "protocol": f"{target['ecode']}_{target['target']}",
                        "name": feat,
                    },
                )
                if not (efeatures.empty):
                    present_features += 1

        # Check if currents
        efeatures = self.db.fetch(
            table="extraction_efeatures",
            conditions={
                "emodel": self.emodel,
                "species": self.species,
                "protocol": "global",
            },
        )
        flag_hypamp = False
        flag_thresh = False
        for efeat in efeatures.to_dict(orient="records"):
            if efeat["name"] == "hypamp":
                flag_hypamp = True
            elif efeat["name"] == "thresh":
                flag_thresh = True

        return flag_hypamp and flag_thresh and present_features / tot_features > 0.8


class OptimisationTarget(luigi.Target):

    """
    Target to check if an optimisation is present in the postgreSQL database.
    """

    def __init__(
        self,
        emodel,
        species,
        optimizer,
        seed,
    ):
        """
        Args:
            emodel (str): name of the emodel.
            species (str): name of the species (rat, human, mouse).
            optimizer (str): algorithm used for optimization, can be "IBEA",
            "SO-CMA", "MO-CMA"
            seed (int): random seed used for optimzation
        """

        self.db = PostgreSQL_API()

        self.emodel = emodel
        self.species = species
        self.optimizer = optimizer
        self.seed = seed

    def exists(self):
        """
        Check if the model is present in the table.
        """

        models = self.db.fetch(
            "models",
            {
                "emodel": self.emodel,
                "species": self.species,
                "optimizer": self.optimizer,
                "seed": self.seed,
            },
        )

        if models.empty:
            return False
        return True


class Task_extract_efeatures(luigi.Task):
    """Luigi wrapper for emodel_pipeline.emodel_creation.extract_efeatures"""

    emodel = luigi.Parameter()
    species = luigi.Parameter()
    db_api = luigi.Parameter(default="sql")
    file_format = luigi.Parameter(default="axon")
    threshold_nvalue_save = luigi.IntParameter(default=1)

    def output(self):
        return EfeaturesProtocolsTarget(
            emodel=self.emodel,
            species=self.species,
        )

    def run(self):
        extract_efeatures(
            emodel=self.emodel,
            species=self.species,
            db_api=self.db_api,
            file_format=self.file_format,
            threshold_nvalue_save=self.threshold_nvalue_save,
        )


class Task_optimize(luigi.Task):
    """Luigi wrapper for emodel_pipeline.emodel_creation.optimize"""

    emodel = luigi.Parameter()
    species = luigi.Parameter()
    db_api = luigi.Parameter(default="sql")
    file_format = luigi.Parameter(default="axon")
    threshold_nvalue_save = luigi.IntParameter(default=1)

    working_dir = luigi.Parameter()
    mechanisms_dir = luigi.Parameter()
    stochasticity = luigi.BoolParameter(default=False)
    copy_mechanisms = luigi.BoolParameter(default=True)
    opt_params = luigi.DictParameter()
    checkpoint_path = luigi.Parameter()
    optimizer = luigi.Parameter()

    def requires(self):
        return Task_extract_efeatures(
            emodel=self.emodel,
            species=self.species,
            db_api=self.db_api,
            file_format=self.file_format,
            threshold_nvalue_save=self.threshold_nvalue_save,
        )

    def output(self):

        if "seed" in self.opt_params:
            seed = self.opt_params["seed"]
        else:
            seed = 1

        return OptimisationTarget(
            emodel=self.emodel,
            species=self.species,
            optimizer=self.optimizer,
            seed=seed,
        )

    def run(self):
        optimize(
            emodel=self.emodel,
            species=self.species,
            db_api=self.db_api,
            working_dir=self.working_dir,
            mechanisms_dir=self.mechanisms_dir,
            stochasticity=self.stochasticity,
            copy_mechanisms=self.copy_mechanisms,
            opt_params=self.opt_params,
            optimizer=self.optimizer,
            checkpoint_path=self.checkpoint_path,
        )
