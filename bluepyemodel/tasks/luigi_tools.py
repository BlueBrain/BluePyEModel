"""Luigi tool module."""
import json
import logging
from abc import ABC

import luigi
from bbp_workflow.task import IPyParallelExclusive
from luigi.parameter import MissingParameterException
from luigi.parameter import _no_value
from luigi_tools.task import _no_default_value

from bluepyemodel import access_point
from bluepyemodel.tasks.config import EmodelAPIConfig
from bluepyemodel.tools.multiprocessing import get_mapper

logger = logging.getLogger(__name__)


class WorkflowTask(luigi.Task):
    """Workflow task with loaded data access point."""

    backend = luigi.Parameter(default=None, config_path=dict(section="parallel", name="backend"))
    emodel = luigi.Parameter()
    ttype = luigi.Parameter()
    iteration_tag = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

        self.access_point = access_point.get_access_point(
            EmodelAPIConfig().api,
            emodel=self.emodel,
            ttype=self.ttype,
            iteration_tag=self.iteration_tag,
            **EmodelAPIConfig().api_args
        )

    def get_mapper(self):
        """Get a mapper for parallel computations."""
        return get_mapper(self.backend)


class WorkflowTaskRequiringMechanisms(WorkflowTask):
    """Workflow task with data access point and download of missing mechanisms"""

    def __init__(self, *args, **kwargs):
        """ """

        super().__init__(*args, **kwargs)

        self.access_point.get_model_configuration()


class WorkflowTarget(luigi.Target, ABC):
    """Workflow target with loaded access_point."""

    def __init__(self, emodel, ttype, iteration_tag, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

        self.emodel = emodel
        self.ttype = ttype
        self.iteration_tag = iteration_tag

        self.access_point = access_point.get_access_point(
            EmodelAPIConfig().api,
            emodel=self.emodel,
            ttype=self.ttype,
            iteration_tag=self.iteration_tag,
            **EmodelAPIConfig().api_args
        )


class WorkflowWrapperTask(WorkflowTask, luigi.WrapperTask):
    """Base wrapper class with global parameters."""


class IPyParallelTask(IPyParallelExclusive):
    """Wrapper around IPyParallelTask to get chdir param from [DEFAULT] in config."""

    chdir = luigi.configuration.get_config().get("DEFAULT", "chdir")

    def prepare_args_for_remote_script(self, attrs):
        """Prepare self.args, which is used to pass arguments to remote_script."""
        # start with '--' to separate ipython arguments from parsing arguments
        self.args = "--"

        for attr in attrs:
            if hasattr(self, attr):

                # Luigi stores lists as tuples, but json cannot load tuple
                # so here, turning tuples back into lists.
                # Also turn lists and dicts into json strings.
                if isinstance(getattr(self, attr), tuple):
                    setattr(self, attr, json.dumps(list(getattr(self, attr))))
                # luigi stores dicts as luigi.freezing.FrozenOrderedDict
                # that are not json serializable,
                # so turn them into dict, and then into json strings
                elif isinstance(getattr(self, attr), (dict, luigi.freezing.FrozenOrderedDict)):
                    setattr(self, attr, json.dumps(dict(getattr(self, attr))))

                if getattr(self, attr) is True:
                    self.args = " ".join([self.args, "--" + attr])
                elif getattr(self, attr) is not False and getattr(self, attr) is not None:
                    # be sure that lists and dicts are inside ' '
                    # so that argparse detect them as one argument
                    # have to change ' to '\\'' because args would already be
                    # inside quotes (') in command from bbp-workflow
                    # single quotes would mess that up
                    self.args = " ".join(
                        [self.args, "--" + attr, "'\\''" + str(getattr(self, attr)) + "'\\''"]
                    )

        # append API-related arguments
        # api is str
        self.args += " --api_from_config " + EmodelAPIConfig().api
        # api_args is dict
        self.args += (
            " --api_args_from_config "
            + "'\\''"
            + json.dumps(dict(EmodelAPIConfig().api_args))
            + "'\\''"
        )


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
