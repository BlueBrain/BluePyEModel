""" Emodels evaluation functions """

"""
Copyright 2023-2024 Blue Brain Project / EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy
import glob
import logging
from pathlib import Path

import numpy
from bluepyopt.ephys.responses import TimeVoltageResponse

from bluepyemodel.access_point import get_access_point
from bluepyemodel.access_point.local import LocalAccessPoint
from bluepyemodel.evaluation.evaluator import create_evaluator
from bluepyemodel.model import model
from bluepyemodel.tools.mechanisms import compile_mechs_in_emodel_dir
from bluepyemodel.tools.mechanisms import delete_compiled_mechanisms
from bluepyemodel.tools.utils import make_dir

logger = logging.getLogger(__name__)


def locally_store_responses(emodel):
    """Locally store the responses.

    Arguments:
        emodel (EModel): the emodel which responses are to be stored
    """
    output_dir = f"./recordings/{emodel.emodel_metadata.as_string(emodel.seed)}"
    make_dir(output_dir)
    for key, resp in emodel.responses.items():
        output_path = Path(output_dir) / ".".join((key, "dat"))
        if not ("holding_current" in key or "threshold_current" in key or "bpo" in key):
            if resp["time"] is not None and resp["voltage"] is not None:
                time = numpy.array(resp["time"])
                data = numpy.array(resp["voltage"])  # even current will be named voltage here

                numpy.savetxt(output_path, numpy.transpose(numpy.vstack((time, data))))
        else:
            numpy.savetxt(output_path, numpy.asarray([resp]))


def check_local_responses_presence(emodels, cell_eval):
    """Returns True if there is a local response file for each emodel.

    Arguments:
        emodels (list of EModel): the emodel that need a response to be checked.
        cell_eval (CellEvaluator): evaluator for the cell model/protocols/e-feature set.
    """
    for emodel in emodels:
        output_dir = f"./recordings/{emodel.emodel_metadata.as_string(emodel.seed)}"
        if not Path(output_dir).is_dir():
            return False

        # only check voltage, since some currents are not recorded
        # because non existent depending on location
        if not all(
            (
                (Path(output_dir) / ".".join((rec.name, "dat"))).is_file()
                for prot in cell_eval.fitness_protocols["main_protocol"].protocols.values()
                for rec in prot.recordings
                if rec.variable == "v"
            )
        ):
            return False

    return True


def load_responses_from_local_files(emodels, cell_eval):
    """Returns responses from locally stored files.

    Arguments:
        emodels (list of EModel): the emodel that need a response to be checked.
        cell_eval (CellEvaluator): evaluator for the cell model/protocols/e-feature set.
    """
    responses_list = []
    for emodel in emodels:
        responses = {}
        output_dir = f"./recordings/{emodel.emodel_metadata.as_string(emodel.seed)}"

        for filepath in glob.glob(str(Path(output_dir) / "*.dat")):
            response_key = Path(filepath).stem
            response_name = response_key.split(".")[0]
            data = numpy.loadtxt(filepath)
            if not (
                "holding_current" in response_key
                or "threshold_current" in response_key
                or "bpo" in response_key
            ):
                responses[response_key] = TimeVoltageResponse(
                    name=response_name, time=data[:, 0], voltage=data[:, 1]
                )
            else:
                responses[response_key] = data
        responses["evaluator"] = copy.deepcopy(cell_eval)

        responses_list.append(responses)

    return responses_list


def get_responses(to_run):
    """Compute the voltage responses of a set of parameters.

    Args:
        to_run (dict): of the form
            to_run = {"evaluator": CellEvaluator, "parameters": Dict}
    """

    eva = to_run["evaluator"]
    params = to_run["parameters"]

    eva.cell_model.unfreeze(params)

    responses = eva.run_protocols(protocols=eva.fitness_protocols.values(), param_values=params)
    responses["evaluator"] = eva

    return responses


def compute_responses(
    access_point,
    cell_evaluator,
    map_function,
    seeds=None,
    preselect_for_validation=False,
    store_responses=False,
    load_from_local=False,
):
    """Compute the responses of the emodel to the optimisation and validation protocols.

    Args:
        access_point (DataAccessPoint): API used to access the data.
        emodel (str): name of the emodel. Has to match the name of the emodel
            under which the configuration data are stored.
        cell_evaluator (CellEvaluator): evaluator for the cell model/protocols/e-feature set.
        map_function (map): used to parallelize the evaluation of the
            individual in the population.
        seeds (list): if not None, filter emodels to keep only the ones with these seeds.
        preselect_for_validation (bool): if True,
            only select models that have not been through validation yet.
        store_responses (bool): whether to locally store the responses.
        load_from_local (bool): True to load responses from locally saved recordings.
    Returns:
        emodels (list): list of emodels.
    """

    emodels = access_point.get_emodels()

    if seeds:
        emodels = [model for model in emodels if model.seed in seeds]
    if access_point.emodel_metadata.iteration:
        emodels = [
            model
            for model in emodels
            if model.emodel_metadata.iteration == access_point.emodel_metadata.iteration
        ]
    if preselect_for_validation:
        emodels = [model for model in emodels if model.passed_validation is None]

    if emodels:
        logger.info("In compute_responses, %s emodels found to evaluate.", len(emodels))

        to_run = []
        for mo in emodels:
            to_run.append(
                {
                    "evaluator": copy.deepcopy(cell_evaluator),
                    "parameters": mo.parameters,
                }
            )

        if load_from_local and check_local_responses_presence(emodels, cell_evaluator):
            logger.info(
                "Local responses file found. Loading them from files instead of recomputing them"
            )
            responses = load_responses_from_local_files(emodels, cell_evaluator)
        else:
            responses = list(map_function(get_responses, to_run))

        for mo, r in zip(emodels, responses):
            mo.responses = r
            mo.evaluator = r.pop("evaluator")
            if store_responses and not load_from_local:
                locally_store_responses(mo)

    else:
        logger.warning(
            "In compute_responses, no emodel for %s",
            access_point.emodel_metadata.emodel,
        )

    return emodels


def fill_initial_parameters(evaluator, initial_parameters):
    """Freezes the parameters of the evaluator that are present in the informed parameter set."""
    # pylint: disable=protected-access
    replaced = []

    for p in evaluator.cell_model.params:
        if (
            p in initial_parameters
            and evaluator.cell_model.params[p].bounds is None
            and evaluator.cell_model.params[p]._value is None
        ):
            logger.info(
                "Parameter %s is set to its value from previous emodel: %s",
                evaluator.cell_model.params[p].name,
                initial_parameters[p],
            )
            evaluator.cell_model.params[p]._value = initial_parameters[p]
            evaluator.cell_model.params[p].frozen = True
            replaced.append(evaluator.cell_model.params[p].name)

    evaluator.params = [p for p in evaluator.params if p.name not in replaced]
    evaluator.param_names = [pn for pn in evaluator.param_names if pn not in replaced]


def get_evaluator_from_access_point(
    access_point,
    stochasticity=None,
    include_validation_protocols=False,
    timeout=None,
    use_fixed_dt_recordings=False,
    record_ions_and_currents=False,
):
    """Create an evaluator for the emodel.

    Args:
        access_point (DataAccessPoint): API used to access the database
        stochasticity (bool): should channels behave stochastically if they can.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        timeout (float): duration (in second) after which the evaluation of a
            protocol will be interrupted.
        use_fixed_dt_recordings (bool): used for legacy currentscape
            to record at a fixed dt of 0.1 ms.
        record_ions_and_currents (bool): whether to add the ion and non-specific currents
            and the ionic concentration to the recordings.

    Returns:
        bluepyopt.ephys.evaluators.CellEvaluator
    """

    model_configuration = access_point.get_model_configuration()
    fitness_calculator_configuration = access_point.get_fitness_calculator_configuration(
        record_ions_and_currents=record_ions_and_currents
    )

    if model_configuration.morph_modifiers:
        morph_modifiers = model_configuration.morph_modifiers
    elif access_point.pipeline_settings:
        morph_modifiers = access_point.pipeline_settings.morph_modifiers
    else:
        morph_modifiers = None

    cell_model = model.create_cell_model(
        name=access_point.emodel_metadata.emodel,
        model_configuration=model_configuration,
        morph_modifiers=morph_modifiers,
    )

    mechanisms_directory = access_point.get_mechanisms_directory()
    if isinstance(access_point, LocalAccessPoint):
        if (
            Path.cwd() != access_point.emodel_dir.resolve()
            and access_point.emodel_metadata.iteration
        ):
            delete_compiled_mechanisms()
            if not (access_point.emodel_dir / "x86_64" / "special").is_file():
                compile_mechs_in_emodel_dir(mechanisms_directory)
        else:
            # if x86_64 present in main repo AND mechanisms_directory given to simulator
            # NEURON loads mechanisms twice and crash
            mechanisms_directory = None

    evaluator = create_evaluator(
        cell_model=cell_model,
        fitness_calculator_configuration=fitness_calculator_configuration,
        pipeline_settings=access_point.pipeline_settings,
        stochasticity=stochasticity,
        timeout=timeout,
        include_validation_protocols=include_validation_protocols,
        mechanisms_directory=mechanisms_directory,
        use_fixed_dt_recordings=use_fixed_dt_recordings,
    )

    start_from_emodel = access_point.pipeline_settings.start_from_emodel

    if start_from_emodel is not None:
        access_point_type = "local" if isinstance(access_point, LocalAccessPoint) else "nexus"

        seed = start_from_emodel.pop("seed", None)

        if access_point_type == "local":
            kwargs = {
                "emodel_dir": access_point.emodel_dir,
                "final_path": access_point.final_path,
                "recipes_path": access_point.recipes_path,
            }
        else:
            raise NotImplementedError("start_from_emodel not implemented for Nexus access point")

        starting_access_point = get_access_point(
            access_point=access_point_type, **start_from_emodel, **kwargs
        )

        emodels = starting_access_point.get_emodels()
        if not emodels:
            raise ValueError(
                f"Cannot start optimisation of {access_point.emodel_metadata.emodel} because"
                f" there are no emodels for {start_from_emodel}"
            )

        if seed is None:
            initial_parameters = sorted(emodels, key=lambda x: x.fitness)[0].parameters
        else:
            initial_parameters = next(
                (e.parameters for e in emodels if str(e.seed) == str(seed)), None
            )
            if initial_parameters is None:
                raise ValueError(
                    f"Cannot start optimisation of {access_point.emodel_metadata.emodel} because"
                    f" there are no emodels for {start_from_emodel}"
                )

        fill_initial_parameters(evaluator, initial_parameters)

    return evaluator
