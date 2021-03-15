"""Luigi tool module."""
import logging
import os
from abc import ABC
from contextlib import contextmanager
from pathlib import Path

import luigi
from luigi.parameter import MissingParameterException
from luigi.parameter import _no_value
from luigi_tools.task import WorkflowTask as _WorkflowTask
from luigi_tools.task import _no_default_value

from bluepyemodel import api
from bluepyemodel.tasks.config import EmodelAPIConfig
from bluepyemodel.tasks.utils import change_cwd
from bluepyemodel.tasks.utils import generate_githash
from bluepyemodel.tasks.utils import generate_versions
from bluepyemodel.tasks.utils import update_gitignore

# pylint: disable=W0107

logger = logging.getLogger(__name__)


class WorkflowTask(_WorkflowTask):
    """Workflow task with loaded emodel_db."""

    backend = luigi.Parameter(default=None, config_path=dict(section="parallel", name="backend"))

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.emodel_db = api.get_db(EmodelAPIConfig().api, **EmodelAPIConfig().api_args)

    def get_mapper(self):
        """Get a mapper for parallel computations."""
        if self.backend == "ipyparallel":
            from bluepyemodel.tasks.utils import ipyparallel_map_function

            return ipyparallel_map_function()

        if self.backend == "multiprocessing":
            from bluepyemodel.tasks.utils import NestedPool

            pool = NestedPool()
            return pool.map
        return map

    def on_sucess(self):
        """Close emodel db once we are done. -> Deprecated"""
        pass


class WorkflowTarget(luigi.Target, ABC):
    """Workflow target with loaded emodel_db."""

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.emodel_db = api.get_db(EmodelAPIConfig().api, **EmodelAPIConfig().api_args)


class WorkflowWrapperTask(WorkflowTask, luigi.WrapperTask):
    """Base wrapper class with global parameters."""


class BoolParameterCustom(luigi.BoolParameter):
    """Class to make luigi BoolParameter compatible with luigi-tools's copy-params."""

    def task_value(self, task_name, param_name):
        value = self._get_value(task_name, param_name)
        if value == _no_value:
            raise MissingParameterException("No default specified")
        if value == _no_default_value:
            return _no_default_value
        return self.normalize(value)

    def parse(self, val):
        """
        Parses a ``bool`` from the string, matching 'true' or 'false' ignoring case.
        """
        if val == _no_default_value:
            return _no_default_value
        return super().parse(val)


class ListParameterCustom(luigi.ListParameter):
    """Class to make luigi ListParameter compatible with luigi-tools's copy-params.

    When a class that has copy-params is yielded, this class should replace luigi.ListParameter.
    """

    def parse(self, x):
        """
        Parse an individual value from the input.

        Do not parse if value is None or _no_default_value

        :param str x: the value to parse.
        :return: the parsed value.
        """
        if x is None:
            return None
        if x == _no_default_value:
            return _no_default_value
        return super().parse(x)

    def serialize(self, x):
        """
        Opposite of :py:meth:`parse`.

        Converts the value ``x`` to a string unless x is _no_default_value.

        :param x: the value to serialize.
        """
        if x == _no_default_value:
            return _no_default_value
        return super().serialize(x)


#################
# UNSUSED BELOW #
#################


class UseGitMixin:
    """
    Mixin class for git management.
    """

    @contextmanager
    def use_git_directory(self):
        """If use_git, change temporarily working_dir according to githash."""
        # store original directory
        original_wd = Path.cwd()
        original_working_dir = self.working_dir

        # change directory
        change_cwd(self.working_dir)

        # if git, get githash and change directory
        if self.use_git:

            run_dir = "./run"

            is_git = str(os.popen("git rev-parse --is-inside-work-tree").read())[:-1]
            if is_git != "true":
                raise Exception(
                    "use_git is true, but there is no git repository initialized in working_dir."
                )

            if self.githash is None:

                update_gitignore()
                # generate_version has to be ran before generating the githash as a change
                # in a package version should induce the creation of a new githash
                generate_versions()
                self.githash = generate_githash(run_dir)

            else:
                logger.info("Working from existing githash directory: %s", self.githash)

            working_dir = str(Path(self.working_dir) / run_dir / self.githash)
            change_cwd(working_dir)

        elif self.githash:
            raise Exception("A githash was provided but use_git is False.")

        # since we changed directory, working_dir is ./
        self.working_dir = "./"

        # continue
        yield

        # restore original directory
        change_cwd(original_wd)
        self.working_dir = original_working_dir
