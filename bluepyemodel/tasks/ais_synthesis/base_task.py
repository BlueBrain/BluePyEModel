"""Generic tasks base don luigi.Task."""
from pathlib import Path

import luigi

from luigi_tools.task import WorkflowTask, WorkflowWrapperTask

from bluepyemodel.api.singlecell import Singlecell_API
from .config import EmodelAPIConfig


class EmodelAwareTask:
    """Task with loaded emodel_db."""

    def get_database(self):
        """Fetch emodel AP."""
        if EmodelAPIConfig().api == "singlecell":
            return Singlecell_API(
                working_dir=EmodelAPIConfig().working_dir,
                final_path=EmodelAPIConfig().final_path,
                legacy_dir_structure=True,
            )
        raise NotImplementedError(f"api {EmodelAPIConfig().api} is not implemented")

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.emodel_db = self.get_database()


class BaseTask(EmodelAwareTask, WorkflowTask):
    """Base task of ais_synthesis workflow."""

    continu = luigi.BoolParameter(
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
        # pylint: disable=no-member,protected-access
        return self.output()._prefix / Path(tmp_path) / path


class BaseWrapperTask(EmodelAwareTask, WorkflowWrapperTask):
    """WrapperTask"""
