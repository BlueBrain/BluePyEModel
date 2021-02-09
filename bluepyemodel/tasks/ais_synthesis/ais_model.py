"""Main luigi tasks to run the ais workflow computations."""
import logging
from pathlib import Path

import luigi
import pandas as pd
import yaml

from bluepyemodel.ais_synthesis.ais_model import build_ais_diameter_models
from bluepyemodel.ais_synthesis.ais_model import build_ais_resistance_models
from bluepyemodel.ais_synthesis.ais_model import find_target_rho_axon
from bluepyemodel.ais_synthesis.tools import init_parallel_factory
from bluepyemodel.tasks.ais_synthesis.base_task import BaseTask
from bluepyemodel.tasks.ais_synthesis.config import ModelLocalTarget
from bluepyemodel.tasks.ais_synthesis.config import ScaleConfig
from bluepyemodel.tasks.ais_synthesis.morph_combos import ApplySubstitutionRules
from bluepyemodel.tasks.ais_synthesis.morph_combos import CreateMorphCombosDF
from bluepyemodel.tasks.ais_synthesis.utils import ensure_dir

logger = logging.getLogger(__name__)


class AisShapeModel(BaseTask):
    """Constructs the AIS shape models from data."""

    morphology_path = luigi.Parameter(default="repaired_morphology_path")

    mtypes = luigi.Parameter(default="all")
    mtype_dependent = luigi.BoolParameter(default=False)
    with_taper = luigi.BoolParameter(default=True)
    target_path = luigi.Parameter(default="morphs_combos_df.csv")

    def run(self):
        """Run."""
        morphs_df = pd.read_csv(ApplySubstitutionRules().morphs_df_path)
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
        """"""
        return ModelLocalTarget(self.target_path)


class AisResistanceModel(BaseTask):
    """Constructs the AIS input resistance models."""

    emodel = luigi.Parameter(default=None)
    fit_df_path = luigi.Parameter(default="csv/fit_df.csv")
    fit_db_path = luigi.Parameter(default="sql/fit_db.sql")
    morphology_path = luigi.Parameter(default="repaired_morphology_path")

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
            continu=self.continu,
            combos_db_filename=self.set_tmp(self.add_emodel(self.fit_db_path)),
        )

        ensure_dir(self.set_tmp(self.add_emodel(self.fit_df_path)))
        fit_df.to_csv(self.set_tmp(self.add_emodel(self.fit_df_path)), index=False)

        with self.output().open("w") as f:
            yaml.dump(ais_models, f)

        parallel_factory.shutdown()

    def output(self):
        """"""
        return ModelLocalTarget(self.add_emodel(self.target_path))


class TargetRhoAxon(BaseTask):
    """Estimate the target rho axon value per me-types."""

    emodel = luigi.Parameter(default=None)
    morphology_path = luigi.Parameter(default="repaired_morphology_path")
    custom_target_rhos_path = luigi.Parameter(default="custom_target_rhos.yaml")
    rho_scan_db_path = luigi.Parameter(default="sql/rho_scan_db.sql")
    rho_scan_df_path = luigi.Parameter(default="csv/rho_scan_df.csv")

    target_path = luigi.Parameter(default="target_rhos/target_rhos.yaml")

    def requires(self):
        """Requires."""

        return {
            "ais_model": AisResistanceModel(emodel=self.emodel),
            "morph_combos": CreateMorphCombosDF(),
        }

    def run(self):
        """Run."""
        target_rhos = None
        morphs_combos_df = pd.read_csv(self.input()["morph_combos"].path)

        with self.input()["ais_model"].open() as f:
            ais_models = yaml.full_load(f)

        ensure_dir(self.set_tmp(self.add_emodel(self.rho_scan_db_path)))
        parallel_factory = init_parallel_factory(self.parallel_lib)
        rho_scan_df, target_rhos = find_target_rho_axon(
            morphs_combos_df,
            self.emodel_db,
            self.emodel,
            ais_models,
            ScaleConfig().scales_params,
            morphology_path=self.morphology_path,
            combos_db_filename=self.set_tmp(self.add_emodel(self.rho_scan_db_path)),
            continu=self.continu,
            parallel_factory=parallel_factory,
        )

        ensure_dir(self.set_tmp(self.add_emodel(self.rho_scan_df_path)))
        rho_scan_df.to_csv(self.set_tmp(self.add_emodel(self.rho_scan_df_path)), index=False)

        if Path(self.custom_target_rhos_path).exists():
            custom_target_rho = yaml.full_load(open(self.custom_target_rhos_path, "r"))
            if custom_target_rho is not None and self.emodel in custom_target_rho:
                logger.info("Using custom value for target rho axon")
                target_rhos = {self.emodel: {"all": custom_target_rho[self.emodel]}}

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(target_rhos, f)

        parallel_factory.shutdown()

    def output(self):
        """"""
        return ModelLocalTarget(self.add_emodel(self.target_path))
