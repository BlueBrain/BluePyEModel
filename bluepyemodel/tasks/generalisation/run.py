"""Luigi task to run the ais workflow."""
import luigi

from bluepyemodel.tasks.generalisation.ais_model import AisResistanceModel
from bluepyemodel.tasks.generalisation.ais_model import TargetRhoAxon
from bluepyemodel.tasks.generalisation.ais_synthesis import SynthesizeAis
from bluepyemodel.tasks.generalisation.base_task import BaseWrapperTask
from bluepyemodel.tasks.generalisation.evaluations import EvaluateExemplars
from bluepyemodel.tasks.generalisation.evaluations import EvaluateGeneric
from bluepyemodel.tasks.generalisation.evaluations import EvaluateSynthesis
from bluepyemodel.tasks.generalisation.gather import GatherAisModels
from bluepyemodel.tasks.generalisation.gather import GatherExemplarEvaluations
from bluepyemodel.tasks.generalisation.gather import GatherGenericEvaluations
from bluepyemodel.tasks.generalisation.gather import GatherSynthAis
from bluepyemodel.tasks.generalisation.gather import GatherSynthEvaluations
from bluepyemodel.tasks.generalisation.gather import GatherTargetRhoAxon
from bluepyemodel.tasks.generalisation.plotting import PlotAisResistanceModel
from bluepyemodel.tasks.generalisation.plotting import PlotAisShapeModel
from bluepyemodel.tasks.generalisation.plotting import PlotGenericEvaluation
from bluepyemodel.tasks.generalisation.plotting import PlotGenericSelected
from bluepyemodel.tasks.generalisation.plotting import PlotSelected
from bluepyemodel.tasks.generalisation.plotting import PlotSynthesisEvaluation
from bluepyemodel.tasks.generalisation.plotting import PlotTargetRhoAxon
from bluepyemodel.tasks.generalisation.select import ApplyMegating
from bluepyemodel.tasks.generalisation.select import SelectCombos
from bluepyemodel.tasks.generalisation.select import SelectGenericCombos


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
