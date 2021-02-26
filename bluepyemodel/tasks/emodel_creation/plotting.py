"""Luigi plotting tasks."""

from pathlib import Path

import luigi

from bluepyemodel.emodel_pipeline.plotting import optimization
from bluepyemodel.tasks.emodel_creation.optimisation import Optimize
from bluepyemodel.tasks.luigi_tools import WorkflowTask


class PlotOptimisation(WorkflowTask):
    """Luigi wrapper for plotting the optimisation outputs.

    Parameters:
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        seed (int): seed used in the optimisation.
        checkpoint_dir (str): path to the repo where files used as a checkpoint by BluePyOpt are.
        figures_dir (str): path to figures repo.
    """

    emodel = luigi.Parameter()
    seed = luigi.IntParameter(default=42)
    checkpoint_dir = luigi.Parameter("./checkpoints/")
    figures_dir = luigi.Parameter(default="./figures")

    def requires(self):
        """"""
        return Optimize()

    def run(self):
        """"""
        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_{self.emodel}_{self.seed}.pkl"
        optimization(
            checkpoint_path=checkpoint_path,
            figures_dir=Path(self.figures_dir) / self.emodel,
            emodel=self.emodel,
        )

    def output(self):
        """"""
        figure_filename = f"checkpoint_{self.emodel}_{self.seed}.pdf"
        return luigi.LocalTarget(Path(self.figures_dir) / self.emodel / figure_filename)
