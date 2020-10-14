"""Main luigi tasks to run the ais workflow computations."""
import logging
from pathlib import Path

import pandas as pd
import yaml

import luigi

from ...ais_synthesis.ais_model import (
    build_ais_diameter_models,
    build_ais_resistance_models,
    find_target_rho_axon,
    get_rho_targets,
)
from .base_task import BaseTask
from .utils import (
    add_emodel,
    ensure_dir,
)
from .config import morphologyconfigs, scaleconfigs

logger = logging.getLogger(__name__)


class AisShapeModel(BaseTask):
    """Constructs the AIS shape models from data."""

    morphology_path = luigi.Parameter(default="morphology_path")

    mtypes = luigi.Parameter(default="all")
    mtype_dependent = luigi.BoolParameter(default=False)

    def run(self):
        """Run."""
        morphs_df = pd.read_csv(morphologyconfigs().morphs_df_path)
        morphs_df = morphs_df[morphs_df.use_axon]

        models = build_ais_diameter_models(
            morphs_df,
            self.mtypes,
            morphology_path=self.morphology_path,
            mtype_dependent=self.mtype_dependent,
        )

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(models, f)


class AisResistanceModel(BaseTask):
    """Constructs the AIS input resistance models."""

    emodel = luigi.Parameter()
    fit_df_path = luigi.Parameter(default="fit_df.csv")
    fit_db_path = luigi.Parameter(default="fit_db.sql")
    morphology_path = luigi.Parameter(default="morphology_path")

    def requires(self):
        """Requires."""

        tasks = {"ais_model": AisShapeModel()}
        return self.add_dask_task(tasks)

    def run(self):
        """Run."""

        morphs_combos_df = pd.read_csv(morphologyconfigs().morphs_combos_df_path)

        with self.input()["ais_model"].open() as f:
            ais_models = yaml.full_load(f)

        ensure_dir(self.fit_db_path)
        fit_df, ais_models = build_ais_resistance_models(
            morphs_combos_df,
            self.emodel_db,
            self.emodel,
            ais_models,
            scaleconfigs().scales_params,
            morphology_path=self.morphology_path,
            ipyp_profile=self.ipyp_profile,
            continu=self.continu,
            combos_db_filename=add_emodel(self.fit_db_path, self.emodel),
        )

        ensure_dir(self.fit_df_path)
        fit_df.to_csv(add_emodel(self.fit_df_path, self.emodel))

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(ais_models, f)


class TargetRhoAxon(BaseTask):
    """Estimate the target rho axon value per me-types."""

    emodel = luigi.Parameter()
    morphology_path = luigi.Parameter(default="morphology_path")
    mode = luigi.ChoiceParameter(
        default="fit_cells", choices=["fit_cells", "linear_model"]
    )
    custom_target_rhos_path = luigi.Parameter(default="custom_target_rhos.yaml")
    rho_scan_db_path = luigi.Parameter(default="rho_scan_db.sql")
    rho_scan_df_path = luigi.Parameter(default="rho_scan_df.csv")

    def requires(self):
        """Requires."""

        tasks = {"res_model": AisResistanceModel(emodel=self.emodel)}
        return self.add_dask_task(tasks)

    def run(self):
        """Run."""
        target_rhos = None
        if Path(self.custom_target_rhos_path).exists():
            custom_target_rho = yaml.full_load(open(self.custom_target_rhos_path, "r"))
            if self.emodel in custom_target_rho:
                logger.info("Using custom value for target rho axon")
                target_rhos = {self.emodel: {"all": custom_target_rho[self.emodel]}}

        if not target_rhos and self.mode == "linear_model":
            final = yaml.full_load(open(morphologyconfigs().emodel_path, "r"))
            target_rhos = {
                self.emodel: {"all": float(get_rho_targets(final)[self.emodel])}
            }

        if not target_rhos and self.mode == "fit_cells":
            morphs_combos_df = pd.read_csv(morphologyconfigs().morphs_combos_df_path)

            with self.input()["res_model"].open() as f:
                ais_models = yaml.full_load(f)

            ensure_dir(self.rho_scan_db_path)
            rho_scan_df, target_rhos = find_target_rho_axon(
                morphs_combos_df,
                self.emodel_db,
                self.emodel,
                ais_models,
                scaleconfigs().scales_params,
                morphology_path=self.morphology_path,
                combos_db_filename=add_emodel(self.rho_scan_db_path, self.emodel),
                continu=self.continu,
                ipyp_profile=self.ipyp_profile,
            )

            ensure_dir(self.rho_scan_df_path)
            rho_scan_df.to_csv(add_emodel(self.rho_scan_df_path, self.emodel))

        ensure_dir(self.output().path)
        with self.output().open("w") as f:
            yaml.dump(target_rhos, f)
