"""Cell model creation."""
import collections
import logging
import pathlib
from importlib.machinery import SourceFileLoader

import bluepyopt.ephys as ephys
from bluepyopt.ephys.morphologies import NrnFileMorphology

from .modifiers import replace_axon_hoc
from .modifiers import replace_axon_with_taper

logger = logging.getLogger(__name__)


def multi_locations(section_name, definition):
    """Define a list of locations from a section names.

    Args:
        section_name (str): name of section

    Returns:
        list: list of NrnSeclistLocation

    """
    multiloc_map = {
        "alldend": ["apical", "basal"],
        "somadend": ["apical", "basal", "somatic"],
        "somaxon": ["axonal", "somatic"],
        "allact": ["apical", "basal", "somatic", "axonal"],
    }
    if "multiloc_map" in definition and definition["multiloc_map"] is not None:
        multiloc_map.update(definition["multiloc_map"])

    return [
        ephys.locations.NrnSeclistLocation(sec, seclist_name=sec)
        for sec in multiloc_map.get(section_name, [section_name])
    ]


def define_parameters(definitions):
    """Define a list of NrnParameter from a definition dictionary

    Args:
        definitions (dict): definitions of the parameters and distributions
        used by the model. Of the form:
            definitions = {
                'distributions':
                    {'distrib_name': {
                        'function': function,
                        'parameters': ['param_name']}
                     },
                'parameters':
                    {'sectionlist_name': [
                            {'name': param_name1, 'val': [lbound1, ubound1]},
                            {'name': param_name2, 'val': 3.234}
                        ]
                     }
            }

    Returns:
        list: list of NrnParameter
    """
    # set distributions
    distributions = collections.OrderedDict()
    distributions["uniform"] = ephys.parameterscalers.NrnSegmentLinearScaler()

    distributions_definitions = definitions["distributions"]
    for distribution, definition in distributions_definitions.items():
        distributions[distribution] = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
            name=distribution,
            distribution=definition["fun"],
            dist_param_names=definition.get("parameters", None),
            soma_ref_location=definition.get("soma_ref_location", 0.5),
        )

    params_definitions = definitions["parameters"]
    params_definitions.pop("__comment", None)

    parameters = []
    for sectionlist, params in params_definitions.items():
        dist = None
        seclist_locs = None
        if "distribution_" in sectionlist:
            dist = distributions[sectionlist.split("distribution_")[1]]
        else:
            seclist_locs = multi_locations(sectionlist, definitions)

        for param_config in params:
            param_name = param_config["name"]

            if isinstance(param_config["val"], (list, tuple)):
                is_frozen = False
                value = None
                bounds = param_config["val"]
            else:
                is_frozen = True
                value = param_config["val"]
                bounds = None

            if sectionlist == "global":
                parameters.append(
                    ephys.parameters.NrnGlobalParameter(
                        name=param_name,
                        param_name=param_name,
                        frozen=is_frozen,
                        bounds=bounds,
                        value=value,
                    )
                )
            elif dist:
                parameters.append(
                    ephys.parameters.MetaParameter(
                        name="%s.%s" % (param_name, sectionlist),
                        obj=dist,
                        attr_name=param_name,
                        frozen=is_frozen,
                        bounds=bounds,
                        value=value,
                    )
                )

            elif "dist" in param_config:
                parameters.append(
                    ephys.parameters.NrnRangeParameter(
                        name="%s.%s" % (param_name, sectionlist),
                        param_name=param_name,
                        value_scaler=distributions[param_config["dist"]],
                        value=value,
                        bounds=bounds,
                        frozen=is_frozen,
                        locations=seclist_locs,
                    )
                )
            else:
                parameters.append(
                    ephys.parameters.NrnSectionParameter(
                        name="%s.%s" % (param_name, sectionlist),
                        param_name=param_name,
                        value_scaler=distributions["uniform"],
                        value=value,
                        bounds=bounds,
                        frozen=is_frozen,
                        locations=seclist_locs,
                    )
                )

    return parameters


def define_mechanisms(mechanisms_definition):
    """Define a list of NrnMODMechanism from a definition dictionary

    Args:
        mechanisms_definition (dict): definition of the mechanisms.
            Dictionary of the form:
                mechanisms_definition = {
                    section_name1: {
                        "mech":[
                            mech_name1,
                            mech_name2
                        ]
                    },
                    section_name2: {
                        "mech": [
                            mech_name3,
                            mech_name4
                        ]
                    }
                }

    Returns:
        list: list of NrnMODMechanism
    """
    multiloc_map = {"multiloc_map": mechanisms_definition.get("multiloc_map", None)}
    mechanisms_definition.pop("multiloc_map", None)

    mechanisms = []
    for sectionlist, channels in mechanisms_definition.items():
        seclist_locs = multi_locations(sectionlist, multiloc_map)
        for channel, stoch in zip(channels["mech"], channels["stoch"]):
            mechanisms.append(
                ephys.mechanisms.NrnMODMechanism(
                    name="%s.%s" % (channel, sectionlist),
                    mod_path=None,
                    prefix=channel,
                    locations=seclist_locs,
                    preloaded=True,
                    deterministic=not stoch,
                )
            )

    return mechanisms


def define_morphology(
    morphology_path,
    do_set_nseg=True,
    nseg_frequency=40,
    morph_modifiers=None,
    morph_modifiers_hoc=None,
):
    """Define a morphology object from a morphology file

    Args:
        morphology_path (str): path to a morphology file
        do_set_nseg (float): set the length for the discretization
            of the segments
        nseg_frequency (float): frequency of nseg
        morph_modifiers (list): list of functions to modify the icell
                with (sim, icell) as arguments,
                if None, evaluation.modifiers.replace_axon_with_taper will be used
        morph_modifiers_hoc (list): list of hoc strings corresponding
                to morph_modifiers, each modifier can be a function, or a list of a path
                to a .py and the name of the function to use in this file

    Returns:
        bluepyopt.ephys.morphologies.NrnFileMorphology: a morphology object
    """

    if morph_modifiers is None:
        morph_modifiers = [replace_axon_with_taper]
        morph_modifiers_hoc = [replace_axon_hoc]  # TODO: check the hoc is correct
        logger.warning("No morphology modifiers provided, replace_axon_with_taper will be used.")
    else:
        for i, morph_modifier in enumerate(morph_modifiers):
            if isinstance(morph_modifier, list):
                # pylint: disable=deprecated-method,no-value-for-parameter
                modifier_module = SourceFileLoader(
                    pathlib.Path(morph_modifier[0]).stem, morph_modifier[0]
                ).load_module()
                morph_modifiers[i] = getattr(modifier_module, morph_modifier[1])

            elif not callable(morph_modifier):
                raise Exception("A morph modifier is not callable nor a list of two str")

    return NrnFileMorphology(
        morphology_path,
        do_replace_axon=False,
        do_set_nseg=do_set_nseg,
        nseg_frequency=nseg_frequency,
        morph_modifiers=morph_modifiers,
        morph_modifiers_hoc=morph_modifiers_hoc,
    )


def create_cell_model(
    name,
    morphology,
    mechanisms,
    parameters,
    morph_modifiers=None,
    morph_modifiers_hoc=None,
    seclist_names=None,
    secarray_names=None,
):
    """Create a cell model based on a morphology, mechanisms and parameters

    Args:
        name (str): name of the model
        morphology (dict): morphology from emodel api .get_morphologies()
        mechanisms (dict): see docstring of function define_mechanisms for the
            format
        parameters (dict):  see docstring of function define_parameters for the
            format
        morph_modifiers (list): list of functions to modify morphologies
        morph_modifiers_hoc (list): list of hoc functions to modify morphologies

    Returns:
        CellModel
    """

    morph = define_morphology(
        morphology["path"],
        do_set_nseg=True,
        nseg_frequency=40,
        morph_modifiers=morph_modifiers,
        morph_modifiers_hoc=morph_modifiers_hoc,
    )

    if seclist_names is None:
        seclist_names = morphology.get("seclist_names", None)
    if secarray_names is None:
        secarray_names = morphology.get("secarray_names", None)

    return ephys.models.CellModel(
        name=name,
        morph=morph,
        mechs=define_mechanisms(mechanisms),
        params=define_parameters(parameters),
        seclist_names=seclist_names,
        secarray_names=secarray_names,
    )
