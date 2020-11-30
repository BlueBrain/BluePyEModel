"""Luigi task to run the ais workflow."""
import luigi

from .base_task import BaseWrapperTask
from .ais_model import AisResistanceModel, TargetRhoAxon
from .ais_synthesis import SynthesizeAis
from .evaluations import EvaluateGeneric, EvaluateExemplars, EvaluateSynthesis
from .gather import (
    GatherAisModels,
    GatherGenericEvaluations,
    GatherSynthAis,
    GatherSynthEvaluations,
    GatherTargetRhoAxon,
    GatherExemplarEvaluations,
)
from .select import ApplyMegating, SelectCombos
from .plotting import (
    PlotAisResistanceModel,
    PlotAisShapeModel,
    PlotGenericEvaluation,
    PlotSynthesisEvaluation,
    PlotTargetRhoAxon,
    PlotSelected,
    PlotGenericSelected,
)
from .select import SelectGenericCombos


class RunAll(BaseWrapperTask):
    """Main task to run the workflow."""

    emodels = luigi.ListParameter(default=None)
    rerun_emodels = luigi.ListParameter(default=None)

    def requires(self):
        """"""
        if not self.emodels:
            emodel_db = self.get_database()
            self.emodels = list(emodel_db.get_emodel_names().keys())

        tasks = [PlotAisShapeModel()]
        for emodel in self.emodels:
            rerun_emodel = False
            # pylint: disable=unsupported-membership-test
            if self.rerun_emodels is not None and emodel in self.rerun_emodels:
                rerun_emodel = True
                tasks.append(AisResistanceModel(emodel=emodel, rerun=rerun_emodel))
                tasks.append(TargetRhoAxon(emodel=emodel, rerun=rerun_emodel))
                tasks.append(SynthesizeAis(emodel=emodel, rerun=rerun_emodel))
                tasks.append(EvaluateSynthesis(emodel=emodel, rerun=rerun_emodel))
                tasks.append(EvaluateExemplars(emodel=emodel, rerun=rerun_emodel))

            tasks.append(PlotAisResistanceModel(emodel=emodel, rerun=rerun_emodel))
            tasks.append(PlotTargetRhoAxon(emodel=emodel, rerun=rerun_emodel))
            tasks.append(PlotSynthesisEvaluation(emodel=emodel, rerun=rerun_emodel))

        tasks.append(GatherAisModels(emodels=self.emodels, rerun=rerun_emodel))
        tasks.append(GatherTargetRhoAxon(emodels=self.emodels, rerun=rerun_emodel))
        tasks.append(GatherSynthAis(emodels=self.emodels, rerun=rerun_emodel))
        tasks.append(GatherSynthEvaluations(emodels=self.emodels, rerun=rerun_emodel))
        tasks.append(GatherExemplarEvaluations(emodels=self.emodels, rerun=rerun_emodel))
        tasks.append(ApplyMegating(emodels=self.emodels, rerun=rerun_emodel))
        tasks.append(SelectCombos(emodels=self.emodels, rerun=rerun_emodel))
        tasks.append(PlotSelected(emodels=self.emodels, rerun=rerun_emodel))
        return tasks


class RunGenericEvaluations(BaseWrapperTask):
    """Main task to run the evaluation of emodels."""

    emodels = luigi.ListParameter(default=None)

    def requires(self):
        """"""
        if not self.emodels:
            emodel_db = self.get_database()
            self.emodels = list(emodel_db.get_emodel_names().keys())

        tasks = [GatherGenericEvaluations(emodels=self.emodels)]
        tasks.append(SelectGenericCombos(emodels=self.emodels))

        for emodel in self.emodels:
            tasks.append(EvaluateGeneric(emodel=emodel))
            tasks.append(PlotGenericEvaluation(emodel=emodel))
        tasks.append(PlotGenericSelected(emodels=self.emodels))
        return tasks
