"""Luigi config classes."""
import luigi
from luigi_tools.target import OutputLocalTarget

from .utils import get_database


class EmodelAPIConfig(luigi.Config):
    """Configuration of emodel api database."""

    api = luigi.Parameter(default="local")
    emodel_dir = luigi.Parameter(default="config")
    final_path = luigi.Parameter(default=None)
    emodels = luigi.ListParameter(default=None)

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        if self.emodels is None:
            emodel_db = get_database(self)
            self.emodels = list(emodel_db.get_emodel_names().keys())


class SelectConfig(luigi.Config):
    """Parameter for select step."""

    megate_thresholds_path = luigi.Parameter(default="megate_thresholds.yaml")


class ScaleConfig(luigi.Config):
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


class PathConfig(luigi.Config):
    """Morphology path configuration."""

    # Output tree
    result_path = luigi.Parameter(default="out", description=":str: Path to the output directory.")

    model_subpath = luigi.Parameter(
        default="models", description=":str: Path to the model subdirectory"
    )

    morphcombo_subpath = luigi.Parameter(
        default="morph_combos", description=":str: Path to the morph combos subdirectory"
    )

    synthesis_subpath = luigi.Parameter(
        default="synthesis", description=":str: Path to the synthesis results subdirectory"
    )

    evaluation_subpath = luigi.Parameter(
        default="evaluations", description=":str: Path to the evaluations results subdirectory"
    )

    gather_subpath = luigi.Parameter(
        default="finals", description=":str: Path to the gathered results subdirectory"
    )

    select_subpath = luigi.Parameter(
        default="select", description=":str: Path to the select results subdirectory"
    )

    plot_subpath = luigi.Parameter(
        default="figures", description=":str: Path to the figures subdirectory"
    )


class ModelLocalTarget(OutputLocalTarget):
    """Specific target for models targets."""


class MorphComboLocalTarget(OutputLocalTarget):
    """Specific target for combos targets."""


class SynthesisLocalTarget(OutputLocalTarget):
    """Specific target for synthesis targets."""


class EvaluationLocalTarget(OutputLocalTarget):
    """Specific target for evaluation targets."""


class GatherLocalTarget(OutputLocalTarget):
    """Specific target for gather targets."""


class SelectLocalTarget(OutputLocalTarget):
    """Specific target for select targets."""


class PlotLocalTarget(OutputLocalTarget):
    """Specific target for plotting targets."""


def reset_default_prefixes():
    """Set default output paths for targets."""
    # pylint: disable=protected-access

    OutputLocalTarget.set_default_prefix(PathConfig().result_path)
    ModelLocalTarget.set_default_prefix(PathConfig().model_subpath)
    MorphComboLocalTarget.set_default_prefix(PathConfig().morphcombo_subpath)
    EvaluationLocalTarget.set_default_prefix(PathConfig().evaluation_subpath)
    SynthesisLocalTarget.set_default_prefix(PathConfig().synthesis_subpath)
    GatherLocalTarget.set_default_prefix(PathConfig().gather_subpath)
    SelectLocalTarget.set_default_prefix(PathConfig().select_subpath)
    PlotLocalTarget.set_default_prefix(PathConfig().plot_subpath)


reset_default_prefixes()
