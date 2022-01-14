"""Luigi task to run the ais workflow."""
import luigi

from bluepyemodel.tasks.generalisation.ais_model import AisResistanceModel
from bluepyemodel.tasks.generalisation.ais_synthesis import SynthesizeAisSoma
from bluepyemodel.tasks.generalisation.base_task import BaseWrapperTask
from bluepyemodel.tasks.generalisation.config import EmodelAPIConfig
from bluepyemodel.tasks.generalisation.evaluations import EvaluateExemplars
from bluepyemodel.tasks.generalisation.evaluations import EvaluateGeneric
from bluepyemodel.tasks.generalisation.evaluations import EvaluateSynthesis
from bluepyemodel.tasks.generalisation.gather import GatherAisModels
from bluepyemodel.tasks.generalisation.gather import GatherExemplarEvaluations
from bluepyemodel.tasks.generalisation.gather import GatherGenericEvaluations
from bluepyemodel.tasks.generalisation.gather import GatherSynthAis
from bluepyemodel.tasks.generalisation.gather import GatherSynthEvaluations
from bluepyemodel.tasks.generalisation.gather import GatherTargetRho
from bluepyemodel.tasks.generalisation.gather import GatherTargetRhoAxon
from bluepyemodel.tasks.generalisation.plotting import PlotAisResistanceModel
from bluepyemodel.tasks.generalisation.plotting import PlotAisShapeModel
from bluepyemodel.tasks.generalisation.plotting import PlotGenericEvaluation
from bluepyemodel.tasks.generalisation.plotting import PlotGenericSelected
from bluepyemodel.tasks.generalisation.plotting import PlotSelected
from bluepyemodel.tasks.generalisation.plotting import PlotSomaResistanceModel
from bluepyemodel.tasks.generalisation.plotting import PlotSomaShapeModel
from bluepyemodel.tasks.generalisation.plotting import PlotSurfaceComparison
from bluepyemodel.tasks.generalisation.plotting import PlotSynthesisEvaluation
from bluepyemodel.tasks.generalisation.plotting import PlotTargetRhoAxon
from bluepyemodel.tasks.generalisation.select import ApplyMegating
from bluepyemodel.tasks.generalisation.select import SelectCombos
from bluepyemodel.tasks.generalisation.select import SelectGenericCombos


class RunAll(BaseWrapperTask):
    """Main task to run the workflow."""

    rerun_emodels = luigi.ListParameter(default=None)

    def requires(self):
        """ """
        tasks = [PlotAisShapeModel(), PlotSomaShapeModel()]
        for emodel in EmodelAPIConfig().emodels:
            rerun_emodel = False
            # pylint: disable=unsupported-membership-test
            if self.rerun_emodels is not None and emodel in self.rerun_emodels:
                rerun_emodel = True
                tasks.append(AisResistanceModel(emodel=emodel, rerun=rerun_emodel))
                tasks.append(SynthesizeAisSoma(emodel=emodel, rerun=rerun_emodel))
                tasks.append(EvaluateSynthesis(emodel=emodel, rerun=rerun_emodel))
                tasks.append(EvaluateExemplars(emodel=emodel, rerun=rerun_emodel))

            tasks.append(PlotAisResistanceModel(emodel=emodel, rerun=rerun_emodel))
            tasks.append(PlotSomaResistanceModel(emodel=emodel, rerun=rerun_emodel))
            tasks.append(PlotTargetRhoAxon(emodel=emodel, rerun=rerun_emodel))
            tasks.append(PlotSynthesisEvaluation(emodel=emodel, rerun=rerun_emodel))
            tasks.append(PlotSurfaceComparison(emodel=emodel, rerun=rerun_emodel))

        tasks.append(GatherAisModels(emodels=EmodelAPIConfig().emodels, rerun=rerun_emodel))
        tasks.append(GatherTargetRho(emodels=EmodelAPIConfig().emodels, rerun=rerun_emodel))
        tasks.append(GatherTargetRhoAxon(emodels=EmodelAPIConfig().emodels, rerun=rerun_emodel))
        tasks.append(GatherSynthAis(emodels=EmodelAPIConfig().emodels, rerun=rerun_emodel))
        tasks.append(GatherSynthEvaluations(emodels=EmodelAPIConfig().emodels, rerun=rerun_emodel))
        tasks.append(
            GatherExemplarEvaluations(emodels=EmodelAPIConfig().emodels, rerun=rerun_emodel)
        )
        tasks.append(ApplyMegating(emodels=EmodelAPIConfig().emodels, rerun=rerun_emodel))
        tasks.append(SelectCombos(emodels=EmodelAPIConfig().emodels, rerun=rerun_emodel))
        tasks.append(PlotSelected(emodels=EmodelAPIConfig().emodels, rerun=rerun_emodel))
        return tasks


class RunGenericEvaluations(BaseWrapperTask):
    """Main task to run the evaluation of emodels."""

    emodels = luigi.ListParameter(default=None)

    def requires(self):
        """ """
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
