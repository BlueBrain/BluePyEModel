"""Main luigi tasks to run the ais workflow computations."""
import logging
from pathlib import Path

import luigi
import pandas as pd
import yaml
from bluepyparallel import init_parallel_factory

from bluepyemodel.generalisation.ais_model import build_ais_diameter_models
from bluepyemodel.generalisation.ais_model import build_ais_resistance_models
from bluepyemodel.generalisation.ais_model import build_soma_models
from bluepyemodel.generalisation.ais_model import build_soma_resistance_models
from bluepyemodel.generalisation.ais_model import find_target_rho
from bluepyemodel.generalisation.ais_model import find_target_rho_axon
from bluepyemodel.tasks.generalisation.base_task import BaseTask
from bluepyemodel.tasks.generalisation.config import ModelLocalTarget
from bluepyemodel.tasks.generalisation.config import ScaleConfig
from bluepyemodel.tasks.generalisation.morph_combos import CreateMorphCombosDF
from bluepyemodel.tasks.generalisation.utils import ensure_dir

logger = logging.getLogger(__name__)


class SomaShapeModel(BaseTask):
    """Constructs the soma shape models from data."""

    morphology_path = luigi.Parameter(default="morphology_path")

    mtypes = luigi.ListParameter(default="all")
    mtype_dependent = luigi.BoolParameter(default=False)
    target_path = luigi.Parameter(default="soma_shape_model.yaml")

    def requires(self):
        """ """
        return CreateMorphCombosDF()

    def run(self):
        """Run."""
        morphs_df = pd.read_csv(self.input().path)
        models = build_soma_models(
            morphs_df,
            self.mtypes,
            morphology_path=self.morphology_path,
            mtype_dependent=self.mtype_dependent,
        )

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(models, f)

    def output(self):
        """ """
        return ModelLocalTarget(self.target_path)


class SomaResistanceModel(BaseTask):
    """Constructs the AIS input resistance models."""

    emodel = luigi.Parameter(default=None)
    fit_df_path = luigi.Parameter(default="csv/fit_soma_df.csv")
    fit_db_path = luigi.Parameter(default="sql/fit_soma_db.sql")
    morphology_path = luigi.Parameter(default="morphology_path")

    target_path = luigi.Parameter(default="soma_resistances/soma_resistance_model.yaml")

    def requires(self):
        """Requires."""

        return {"soma_model": SomaShapeModel(), "morph_combos": CreateMorphCombosDF()}

    def run(self):
        """Run."""

        morphs_combos_df = pd.read_csv(self.input()["morph_combos"].path)
        if self.emodel is not None:
            morphs_combos_df = morphs_combos_df[morphs_combos_df.emodel == self.emodel]

        with self.input()["soma_model"].open() as f:
            soma_models = yaml.full_load(f)

        ensure_dir(self.set_tmp(self.add_emodel(self.fit_db_path)))
        parallel_factory = init_parallel_factory(self.parallel_lib)
        fit_df, soma_models = build_soma_resistance_models(
            morphs_combos_df,
            self.emodel_db,
            self.emodel,
            soma_models,
            ScaleConfig().scales_params,
            morphology_path=self.morphology_path,
            parallel_factory=parallel_factory,
            resume=self.resume,
            db_url=self.set_tmp(self.add_emodel(self.fit_db_path)),
        )

        ensure_dir(self.set_tmp(self.add_emodel(self.fit_df_path)))
        fit_df.to_csv(self.set_tmp(self.add_emodel(self.fit_df_path)), index=False)

        with self.output().open("w") as f:
            yaml.dump(soma_models, f)

        parallel_factory.shutdown()

    def output(self):
        """ """
        return ModelLocalTarget(self.add_emodel(self.target_path))


class TargetRho(BaseTask):
    """Estimate the target rho value per me-types."""

    emodel = luigi.Parameter(default=None)
    morphology_path = luigi.Parameter(default="morphology_path")
    custom_target_rhos_path = luigi.Parameter(default="custom_target_rhos.yaml")
    rho_scan_db_path = luigi.Parameter(default="sql/rho_scan_db.sql")
    rho_scan_df_path = luigi.Parameter(default="csv/rho_scan_df.csv")

    target_path = luigi.Parameter(default="target_rhos/target_rhos.yaml")

    def requires(self):
        """Requires."""

        return CreateMorphCombosDF()

    def run(self):
        """Run."""
        target_rhos = None
        morphs_combos_df = pd.read_csv(self.input().path)

        ensure_dir(self.set_tmp(self.add_emodel(self.rho_scan_db_path)))
        parallel_factory = init_parallel_factory(self.parallel_lib)
        target_rhos = find_target_rho(
            morphs_combos_df[morphs_combos_df.emodel == self.emodel],
            self.emodel_db,
            self.emodel,
            morphology_path=self.morphology_path,
            resume=self.resume,
            parallel_factory=parallel_factory,
        )

        print("target rhos:", target_rhos)
        ensure_dir(self.set_tmp(self.add_emodel(self.rho_scan_df_path)))
        # rho_scan_df.to_csv(self.set_tmp(self.add_emodel(self.rho_scan_df_path)), index=False)

        if Path(self.custom_target_rhos_path).exists():
            with open(self.custom_target_rhos_path, "r") as custom_target_file:
                custom_target_rho = yaml.full_load(custom_target_file)
            if custom_target_rho is not None and self.emodel in custom_target_rho:
                logger.info("Using custom value for target rho")
                target_rhos = {self.emodel: {"all": custom_target_rho[self.emodel]}}

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(target_rhos, f)

        parallel_factory.shutdown()

    def output(self):
        """ """
        return ModelLocalTarget(self.add_emodel(self.target_path))


class AisShapeModel(BaseTask):
    """Constructs the AIS shape models from data."""

    morphology_path = luigi.Parameter(default="morphology_path")

    mtypes = luigi.Parameter(default="all")
    mtype_dependent = luigi.BoolParameter(default=False)
    with_taper = luigi.BoolParameter(default=True)
    target_path = luigi.Parameter(default="ais_shape_model.yaml")

    def requires(self):
        """ """
        return CreateMorphCombosDF()

    def run(self):
        """Run."""
        morphs_df = pd.read_csv(self.input().path)
        morphs_df = morphs_df[morphs_df.use_axon]

        models = build_ais_diameter_models(
            morphs_df,
            self.mtypes,
            morphology_path=self.morphology_path,
            mtype_dependent=self.mtype_dependent,
            with_taper=self.with_taper,
        )

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(models, f)

    def output(self):
        """ """
        return ModelLocalTarget(self.target_path)


class AisResistanceModel(BaseTask):
    """Constructs the AIS input resistance models."""

    emodel = luigi.Parameter(default=None)
    fit_df_path = luigi.Parameter(default="csv/fit_AIS_df.csv")
    fit_db_path = luigi.Parameter(default="sql/fit_AIS_db.sql")
    morphology_path = luigi.Parameter(default="morphology_path")

    target_path = luigi.Parameter(default="ais_resistances/ais_resistance_model.yaml")

    def requires(self):
        """Requires."""

        return {"ais_model": AisShapeModel(), "morph_combos": CreateMorphCombosDF()}

    def run(self):
        """Run."""

        morphs_combos_df = pd.read_csv(self.input()["morph_combos"].path)
        if self.emodel is not None:
            morphs_combos_df = morphs_combos_df[morphs_combos_df.emodel == self.emodel]

        with self.input()["ais_model"].open() as f:
            ais_models = yaml.full_load(f)

        ensure_dir(self.set_tmp(self.add_emodel(self.fit_db_path)))
        parallel_factory = init_parallel_factory(self.parallel_lib)
        fit_df, ais_models = build_ais_resistance_models(
            morphs_combos_df,
            self.emodel_db,
            self.emodel,
            ais_models,
            ScaleConfig().scales_params,
            morphology_path=self.morphology_path,
            parallel_factory=parallel_factory,
            resume=self.resume,
            db_url=self.set_tmp(self.add_emodel(self.fit_db_path)),
        )

        ensure_dir(self.set_tmp(self.add_emodel(self.fit_df_path)))
        fit_df.to_csv(self.set_tmp(self.add_emodel(self.fit_df_path)), index=False)

        with self.output().open("w") as f:
            yaml.dump(ais_models, f)

        parallel_factory.shutdown()

    def output(self):
        """ """
        return ModelLocalTarget(self.add_emodel(self.target_path))


class TargetRhoAxon(BaseTask):
    """Estimate the target rho axon value per me-types."""

    emodel = luigi.Parameter(default=None)
    morphology_path = luigi.Parameter(default="morphology_path")
    custom_target_rho_axons_path = luigi.Parameter(default="custom_target_rho_axons.yaml")
    rho_scan_db_path = luigi.Parameter(default="sql/rho_axon_scan_db.sql")
    rho_scan_df_path = luigi.Parameter(default="csv/rho_axon_scan_df.csv")

    target_path = luigi.Parameter(default="target_rhos/target_rho_axons.yaml")

    def requires(self):
        """Requires."""

        return {
            "ais_model": AisResistanceModel(emodel=self.emodel),
            "morph_combos": CreateMorphCombosDF(),
        }

    def run(self):
        """Run."""
        target_rho_axons = None
        morphs_combos_df = pd.read_csv(self.input()["morph_combos"].path)

        with self.input()["ais_model"].open() as f:
            ais_models = yaml.full_load(f)

        ensure_dir(self.set_tmp(self.add_emodel(self.rho_scan_db_path)))
        parallel_factory = init_parallel_factory(self.parallel_lib)
        target_rho_axons = find_target_rho_axon(
            morphs_combos_df,
            self.emodel_db,
            self.emodel,
            ais_models,
            ScaleConfig().scales_params,
            morphology_path=self.morphology_path,
            db_url=self.set_tmp(self.add_emodel(self.rho_scan_db_path)),
            resume=self.resume,
            parallel_factory=parallel_factory,
        )

        print("target rho axon:", target_rho_axons)
        ensure_dir(self.set_tmp(self.add_emodel(self.rho_scan_df_path)))
        # rho_scan_df.to_csv(self.set_tmp(self.add_emodel(self.rho_scan_df_path)), index=False)

        if Path(self.custom_target_rho_axons_path).exists():
            with open(self.custom_target_rho_axons_path, "r") as custom_target_file:
                custom_target_rho_axon = yaml.full_load(custom_target_file)
            if custom_target_rho_axon is not None and self.emodel in custom_target_rho_axon:
                logger.info("Using custom value for target rho axon")
                target_rho_axons = {self.emodel: {"all": custom_target_rho_axon[self.emodel]}}

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(target_rho_axons, f)

        parallel_factory.shutdown()

    def output(self):
        """ """
        return ModelLocalTarget(self.add_emodel(self.target_path))
