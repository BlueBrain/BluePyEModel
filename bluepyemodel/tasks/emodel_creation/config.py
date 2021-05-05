"""Configurations for luigi tasks."""

import luigi

from bluepyemodel.tasks.luigi_tools import BoolParameterCustom
from bluepyemodel.tasks.luigi_tools import ListParameterCustom


class OptimizeConfig(luigi.Config):
    """Parameters used in Optimize.

    Parameters:
        mechanisms_dir (str): path of the directory in which the mechanisms
            will be copied and/or compiled. It has to be a subdirectory of
            working_dir.
        morphology_modifiers (list): list of python functions that will be
            applied to all the morphologies.
        max_ngen (int): maximum number of generations of the evolutionary process.
        stochasticity (bool): should channels behave stochastically if they can.
        copy_mechanisms (bool): should the mod files be copied in the local
            mechanisms_dir directory.
        compile_mechanisms (bool): should the mod files be compiled.
        opt_params (dict): optimisation parameters. Keys have to match the
            optimizer's call.
        optimizer (str): algorithm used for optimization, can be "IBEA", "SO-CMA",
            "MO-CMA".
        checkpoint_dir (str): path to the repo where files used as a checkpoint by BluePyOpt are.
        timeout (float): duration (in second) after which the evaluation of a
            protocol will be interrupted.
    """

    mechanisms_dir = luigi.Parameter(default="mechanisms")
    morphology_modifiers = ListParameterCustom(default=None)
    max_ngen = luigi.IntParameter(default=1000)
    stochasticity = BoolParameterCustom(default=False)
    copy_mechanisms = BoolParameterCustom(default=False)
    compile_mechanisms = BoolParameterCustom(default=False)
    opt_params = luigi.DictParameter(default=None)
    optimizer = luigi.Parameter(default="MO-CMA")
    checkpoint_dir = luigi.Parameter("./checkpoints/")
    timeout = luigi.IntParameter(default=600)
