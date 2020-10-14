"""Luigi task to run the ais workflow."""
import pandas as pd

import luigi

from .gather import (
    GatherAisModels,
    GatherGenericEvaluations,
    GatherSynthAis,
    GatherSynthEvaluations,
    GatherTargetRhoAxon,
)
from .evaluations import (
    EvaluateSynthesis,
    SynthesizeAis,
)

from .ais_model import (
    AisResistanceModel,
    TargetRhoAxon,
)
from .plotting import (
    PlotAisResistanceModel,
    PlotAisShapeModel,
    PlotGenericEvaluation,
    PlotSynthesisEvaluation,
    PlotTargetRhoAxon,
)
from .select import (
    PlotGenericSelected,
    PlotSelected,
    SelectCombos,
    SelectGenericCombos,
)
from .config import morphologyconfigs


class RunPlotting(luigi.WrapperTask):
    """Collect all the plotting tasks."""

    emodels = luigi.ListParameter(default=["all"])

    def requires(self):
        """Requires."""
        tasks = [PlotAisShapeModel()]
        for emodel in self.emodels:
            tasks.append(PlotAisResistanceModel(emodel=emodel))
            tasks.append(PlotTargetRhoAxon(emodel=emodel))
            tasks.append(PlotSynthesisEvaluation(emodel=emodel))
        tasks.append(PlotSelected(emodels=self.emodels))

        return tasks


class RunAll(luigi.WrapperTask):
    """Main task to run the workflow."""

    emodels = luigi.ListParameter(default=["all"])
    with_plots = luigi.BoolParameter()

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        if self.emodels[0] == "all":  # pylint: disable=unsubscriptable-object
            self.all_emodels = sorted(
                list(set(pd.read_csv(morphologyconfigs().morphs_combos_df_path).emodel))
            )
        else:
            self.all_emodels = self.emodels

    def requires(self):
        """Requires."""
        tasks = []
        for emodel in self.all_emodels:
            tasks += [
                AisResistanceModel(emodel=emodel),
                TargetRhoAxon(emodel=emodel),
                SynthesizeAis(emodel=emodel),
                EvaluateSynthesis(emodel=emodel),
            ]

        tasks += [
            GatherAisModels(emodels=self.all_emodels),
            GatherTargetRhoAxon(emodels=self.all_emodels),
            GatherSynthAis(emodels=self.all_emodels),
            GatherSynthEvaluations(emodels=self.all_emodels),
            SelectCombos(emodels=self.all_emodels),
        ]

        if self.with_plots:
            tasks.append(RunPlotting(emodels=self.all_emodels))

        return tasks


class RunGenericEvaluations(luigi.WrapperTask):
    """Main task to run the evaluation of emodels."""

    emodels = luigi.ListParameter(default=["all"])
    with_plots = luigi.BoolParameter()

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        if self.emodels[0] == "all":  # pylint: disable=unsubscriptable-object
            self.all_emodels = sorted(
                list(set(pd.read_csv(morphologyconfigs().morphs_combos_df_path).emodel))
            )
        else:
            self.all_emodels = self.emodels

    def requires(self):
        """Requires."""

        tasks = []
        tasks.append(GatherGenericEvaluations(emodels=self.all_emodels))
        tasks.append(SelectGenericCombos(emodels=self.all_emodels))

        if self.with_plots:
            for emodel in self.all_emodels:
                tasks.append(PlotGenericEvaluation(emodel=emodel))
            tasks.append(PlotGenericSelected(emodels=self.all_emodels))

        return tasks
