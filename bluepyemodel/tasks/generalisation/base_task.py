"""Generic tasks base don luigi.Task."""
from pathlib import Path

import luigi
from luigi_tools.task import WorkflowTask
from luigi_tools.task import WorkflowWrapperTask

from bluepyemodel.tasks.generalisation.config import EmodelAPIConfig

from .utils import get_database


class EmodelAwareTask:
    """Task with loaded access_point."""

    def get_database(self):
        """Fetch emodel AP."""
        return get_database(EmodelAPIConfig())

    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

        self.emodel_db = self.get_database()


class BaseTask(EmodelAwareTask, WorkflowTask):
    """Base task of ais_synthesis workflow."""

    resume = luigi.BoolParameter(
        default=False,
        significant=False,
    )
    parallel_lib = luigi.Parameter(
        config_path={"section": "PARALLEL", "name": "parallel_lib"},
        default="multiprocessing",
        significant=False,
    )

    def add_emodel(self, path):
        if hasattr(self, "emodel"):
            # pylint: disable=no-member
            return str(Path(path).with_suffix("")) + "_" + self.emodel + str(Path(path).suffix)
        return path

    def set_tmp(self, path, tmp_path="tmp"):
        """Add tmp_path to hide files in tmp folders."""
        return self.output().super_prefix() / self.output().get_prefix() / Path(tmp_path) / path


class BaseWrapperTask(EmodelAwareTask, WorkflowWrapperTask):
    """WrapperTask"""
