"""Cell model creation."""

"""
Copyright 2023, EPFL/Blue Brain Project

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

import collections
import importlib
import logging

from bluepyopt import ephys
from bluepyopt.ephys.morphologies import NrnFileMorphology
from bluepyopt.ephys.parameterscalers import NrnSegmentLinearScaler
from bluepyopt.ephys.parameterscalers import NrnSegmentSomaDistanceScaler
from bluepyopt.ephys.parameterscalers import NrnSegmentSomaDistanceStepScaler

from bluepyemodel.evaluation import modifiers
from bluepyemodel.evaluation.modifiers import replace_axon_hoc
from bluepyemodel.evaluation.modifiers import replace_axon_with_taper
from bluepyemodel.model.morphology_utils import get_hotspot_location

logger = logging.getLogger(__name__)


def multi_locations(section_name, additional_multiloc_map):
    """Define a list of locations from a section names.

    Args:
        section_name (str): name of section

    Returns:
        list: list of NrnSeclistLocation

    """

    multiloc_map = {
        "alldend": ["apical", "basal"],
        "somadend": ["apical", "basal", "somatic"],
        "allnoaxon": ["apical", "basal", "somatic"],
        "somaxon": ["axonal", "somatic"],
        "allact": ["apical", "basal", "somatic", "axonal"],
    }

    if additional_multiloc_map is not None:
        multiloc_map.update(additional_multiloc_map)

    return [
        ephys.locations.NrnSeclistLocation(sec, seclist_name=sec)
        for sec in multiloc_map.get(section_name, [section_name])
    ]


def define_distributions(distributions_definition, morphology=None):
    """Create a list of ParameterScaler from a the definition of channel distributions

    Args:
        distributions_definition (list): definitions of the distributions
    """
    if any(definition.name == "step" for definition in distributions_definition):
        hotspot_begin, hotspot_end = get_hotspot_location(morphology.morphology_path)
    distributions = collections.OrderedDict()

    for definition in distributions_definition:
        if definition.name == "uniform":
            distributions[definition.name] = NrnSegmentLinearScaler()
        elif definition.name == "step":
            distributions[definition.name] = NrnSegmentSomaDistanceStepScaler(
                name=definition.name,
                distribution=definition.function,
                dist_param_names=definition.parameters,
                soma_ref_location=definition.soma_ref_location,
                step_begin=hotspot_begin,
                step_end=hotspot_end,
            )
        else:
            distributions[definition.name] = NrnSegmentSomaDistanceScaler(
                name=definition.name,
                distribution=definition.function,
                dist_param_names=definition.parameters,
                soma_ref_location=definition.soma_ref_location,
            )

    return distributions


def define_parameters(parameters_definition, distributions, mapping_multilocation):
    """Define a list of NrnParameter from a definition dictionary

    Args:
        parameters_definition (list): definitions of the parameters
        distributions (list): list of distributions in the form of ParameterScaler
        mapping_multilocation (dict): mapping from multi-locations names to list of locations
    """

    parameters = []

    for param_def in parameters_definition:
        if isinstance(param_def.value, (list, tuple)):
            is_frozen = False
            value = None
            bounds = param_def.value
            if bounds[0] > bounds[1]:
                raise ValueError(
                    f"Lower bound ({bounds[0]}) is greater than upper bound ({bounds[1]})"
                    f" for parameter {param_def.name}"
                )
        else:
            is_frozen = True
            value = param_def.value
            bounds = None

        if param_def.location == "global":
            parameters.append(
                ephys.parameters.NrnGlobalParameter(
                    name=param_def.name,
                    param_name=param_def.name,
                    frozen=is_frozen,
                    bounds=bounds,
                    value=value,
                )
            )
            continue

        dist = None
        seclist_locations = None
        if "distribution_" in param_def.location:
            dist = distributions[param_def.location.split("distribution_")[1]]
        else:
            seclist_locations = multi_locations(param_def.location, mapping_multilocation)

        if dist:
            parameters.append(
                ephys.parameters.MetaParameter(
                    name=f"{param_def.name}.{param_def.location}",
                    obj=dist,
                    attr_name=param_def.name,
                    frozen=is_frozen,
                    bounds=bounds,
                    value=value,
                )
            )
        elif param_def.distribution != "uniform":
            parameters.append(
                ephys.parameters.NrnRangeParameter(
                    name=f"{param_def.name}.{param_def.location}",
                    param_name=param_def.name,
                    value_scaler=distributions[param_def.distribution],
                    value=value,
                    bounds=bounds,
                    frozen=is_frozen,
                    locations=seclist_locations,
                )
            )
        else:
            parameters.append(
                ephys.parameters.NrnSectionParameter(
                    name=f"{param_def.name}.{param_def.location}",
                    param_name=param_def.name,
                    value_scaler=distributions["uniform"],
                    value=value,
                    bounds=bounds,
                    frozen=is_frozen,
                    locations=seclist_locations,
                )
            )

    return parameters


def define_mechanisms(mechanisms_definition, mapping_multilocation):
    """Define a list of NrnMODMechanism from a definition dictionary

    Args:
        mechanisms_definition (list of MechanismConfiguration): definition of the mechanisms
        mapping_multilocation (dict): mapping from multi-locations names to list of locations

    Returns:
        list: list of NrnMODMechanism
    """

    mechanisms = []

    for mech_def in mechanisms_definition:
        seclist_locations = multi_locations(mech_def.location, mapping_multilocation)

        mechanisms.append(
            ephys.mechanisms.NrnMODMechanism(
                name=f"{mech_def.name}.{mech_def.location}",
                mod_path=None,
                prefix=mech_def.name,
                locations=seclist_locations,
                preloaded=True,
                deterministic=not mech_def.stochastic,
            )
        )

    return mechanisms


def define_morphology(
    model_configuration,
    do_set_nseg=True,
    nseg_frequency=40,
    morph_modifiers=None,
    morph_modifiers_hoc=None,
):
    """Define a morphology object from a morphology file

    Args:
        model_configuration (NeuronModelConfiguration): configuration of the model
        do_set_nseg (float): set the length for the discretization
            of the segments
        nseg_frequency (float): frequency of nseg
        morph_modifiers (list): list of functions to modify the icell
                with (sim, icell) as arguments,
                if None, evaluation.modifiers.replace_axon_with_taper will be used,
                if ``["bluepyopt_replace_axon"]``, the replace_axon function from
                bluepyopt.ephys.morphologies.NrnFileMorphology will be used
        morph_modifiers_hoc (list): list of hoc strings corresponding
                to morph_modifiers, each modifier can be a function, or a list of a path
                to a .py and the name of the function to use in this file.
                No need to inform them if morph_modifiers is None or "bluepyopt_replace_axon"

    Returns:
        bluepyopt.ephys.morphologies.NrnFileMorphology: a morphology object
    """
    do_replace_axon = False

    if isinstance(morph_modifiers, str):
        morph_modifiers = [morph_modifiers]

    if morph_modifiers is None or morph_modifiers == [None]:
        morph_modifiers = [replace_axon_with_taper]
        morph_modifiers_hoc = [replace_axon_hoc]  # TODO: check the hoc is correct
        logger.debug("No morphology modifiers provided, replace_axon_with_taper will be used.")
    elif morph_modifiers == ["bluepyopt_replace_axon"]:
        morph_modifiers = None
        morph_modifiers_hoc = None
        do_replace_axon = True
    else:
        for i, morph_modifier in enumerate(morph_modifiers):
            if isinstance(morph_modifier, list):
                modifier_module = importlib.import_module(morph_modifier[0])
                morph_modifiers[i] = getattr(modifier_module, morph_modifier[1])
            elif isinstance(morph_modifier, str):
                morph_modifiers[i] = getattr(modifiers, morph_modifier)
            elif not callable(morph_modifier):
                raise TypeError(
                    "A morph modifier is not callable nor a string nor a list of two str"
                )

    return NrnFileMorphology(
        morphology_path=model_configuration.morphology.path,
        do_replace_axon=do_replace_axon,
        do_set_nseg=do_set_nseg,
        nseg_frequency=nseg_frequency,
        morph_modifiers=morph_modifiers,
        morph_modifiers_hoc=morph_modifiers_hoc,
    )


def create_cell_model(
    name,
    model_configuration,
    morph_modifiers=None,
    morph_modifiers_hoc=None,
    seclist_names=None,
    secarray_names=None,
    nseg_frequency=40,
):
    """Create a cell model based on a morphology, mechanisms and parameters

    Args:
        name (str): name of the model
        morphology (dict): morphology from emodel api .get_morphologies()
        model_configuration (NeuronModelConfiguration): Configuration of the neuron model,
            containing the parameters their locations and the associated mechanisms.
        morph_modifiers (list): list of functions to modify morphologies
        morph_modifiers_hoc (list): list of hoc functions to modify morphologies

    Returns:
        CellModel
    """

    morph = define_morphology(
        model_configuration,
        do_set_nseg=True,
        nseg_frequency=nseg_frequency,
        morph_modifiers=morph_modifiers,
        morph_modifiers_hoc=morph_modifiers_hoc,
    )

    if seclist_names is None:
        seclist_names = model_configuration.morphology.seclist_names
    if secarray_names is None:
        secarray_names = model_configuration.morphology.secarray_names

    mechanisms = define_mechanisms(
        model_configuration.mechanisms, model_configuration.mapping_multilocation
    )
    distributions = define_distributions(model_configuration.distributions, morph)
    parameters = define_parameters(
        model_configuration.parameters, distributions, model_configuration.mapping_multilocation
    )

    return ephys.models.CellModel(
        name=name.replace(":", "_").replace("-", "_"),
        morph=morph,
        mechs=mechanisms,
        params=parameters,
        seclist_names=seclist_names,
        secarray_names=secarray_names,
    )
