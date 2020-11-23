"""Luigi task to run the ais workflow."""
import pandas as pd

import luigi

from .base_task import BaseTask
from .evaluations import EvaluateGeneric
from .gather import (
    GatherAisModels,
    GatherGenericEvaluations,
    GatherSynthAis,
    GatherTargetRhoAxon,
)
from .morph_combos import CreateMorphCombosDF
from .plotting import (
    PlotAisResistanceModel,
    PlotAisShapeModel,
    PlotGenericEvaluation,
    PlotSynthesisEvaluation,
    PlotTargetRhoAxon,
)
from .select import PlotGenericSelected, PlotSelected, SelectGenericCombos


class RunAll(BaseTask):
    """Main task to run the workflow."""

    _all_completed = luigi.BoolParameter(default=False)

    def run(self):
        """"""

        morph_combos_task = yield CreateMorphCombosDF()
        all_emodels = list(pd.read_csv(morph_combos_task.path).emodel.unique())

        yield PlotAisShapeModel()

        for emodel in all_emodels:
            yield PlotAisResistanceModel(emodel=emodel)
            yield PlotTargetRhoAxon(emodel=emodel)
            yield PlotSynthesisEvaluation(emodel=emodel)

        yield GatherAisModels(emodels=all_emodels)
        yield GatherTargetRhoAxon(emodels=all_emodels)
        yield GatherSynthAis(emodels=all_emodels)
        yield PlotSelected(emodels=all_emodels)

        self._all_completed = True

    def on_success(self):
        """"""

    def complete(self):
        """"""
        return self._all_completed


class RunGenericEvaluations(BaseTask):
    """Main task to run the evaluation of emodels."""

    _all_completed = luigi.BoolParameter(default=False)

    def run(self):
        """"""
        morph_combos_task = yield CreateMorphCombosDF()
        all_emodels = list(pd.read_csv(morph_combos_task.path).emodel.unique())

        yield GatherGenericEvaluations(emodels=all_emodels)
        yield SelectGenericCombos(emodels=all_emodels)

        for emodel in all_emodels:
            yield EvaluateGeneric(emodel=emodel)
            yield PlotGenericEvaluation(emodel=emodel)
        yield PlotGenericSelected(emodels=all_emodels)

        self._all_completed = True

    def on_success(self):
        """"""

    def complete(self):
        """"""
        return self._all_completed
