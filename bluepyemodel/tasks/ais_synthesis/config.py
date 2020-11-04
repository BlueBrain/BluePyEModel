"""Luigi config classes."""
import luigi


class databaseconfigs(luigi.Config):
    """Configuration of darabase."""

    api = luigi.Parameter(default="singlecell")
    working_dir = luigi.Parameter(default="config")
    final_path = luigi.Parameter(default=None)


class finalconfigs(luigi.Config):
    """Final files paths configs."""

    ais_models_path = luigi.Parameter(default="ais_models.csv")
    target_rhos_path = luigi.Parameter(default="target_rhos.csv")
    synth_combos_df_path = luigi.Parameter(default="synth_combos_df.csv")
    synth_eval_combos_df_path = luigi.Parameter(default="synth_eval_combos_df.csv")
    selected_combos_df_path = luigi.Parameter(default="selected_combos_df.csv")


class selectconfigs(luigi.Config):
    """Parameter for select step."""

    exemplar_morphs_combos_df_path = luigi.Parameter(default="morphs_combos_df.csv")
    exemplar_evaluations_path = luigi.Parameter(default="exemplar_evaluations.csv")
    megated_scores_df_path = luigi.Parameter(default="megated_scores_df.csv")
    megate_thresholds_path = luigi.Parameter(default="megate_thresholds.yaml")


class scaleconfigs(luigi.Config):
    """Scales configuration."""

    scale_min = luigi.FloatParameter(default=-0.8)
    scale_max = luigi.FloatParameter(default=0.8)
    scale_n = luigi.IntParameter(default=50)
    scale_lin = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.scales_params = {
            "min": self.scale_min,
            "max": self.scale_max,
            "n": self.scale_n,
            "lin": self.scale_lin,
        }
