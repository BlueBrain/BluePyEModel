"""Cell model creation."""

import collections
import logging
from pathlib import Path

import bluepyopt.ephys as ephys
from bluepyopt.ephys.morphologies import NrnFileMorphology

from .modifiers import replace_axon_with_taper, replace_axon_hoc

logger = logging.getLogger(__name__)


def multi_locations(section_name):
    """Define a list of locations from a section names.

    Args:
        section_name (str): name of section

    Returns:
        list: list of NrnSeclistLocation

    """

    if section_name == "alldend":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation("basal", seclist_name="basal"),
        ]
    elif section_name == "somadend":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation("basal", seclist_name="basal"),
            ephys.locations.NrnSeclistLocation("somatic", seclist_name="somatic"),
        ]
    elif section_name == "somaxon":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("axonal", seclist_name="axonal"),
            ephys.locations.NrnSeclistLocation("somatic", seclist_name="somatic"),
        ]
    elif section_name == "allact":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation("basal", seclist_name="basal"),
            ephys.locations.NrnSeclistLocation("somatic", seclist_name="somatic"),
            ephys.locations.NrnSeclistLocation("axonal", seclist_name="axonal"),
        ]
    else:
        seclist_locs = [
            ephys.locations.NrnSeclistLocation(section_name, seclist_name=section_name)
        ]

    return seclist_locs


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

    parameters = []

    # set distributions
    distributions = collections.OrderedDict()
    distributions["uniform"] = ephys.parameterscalers.NrnSegmentLinearScaler()

    distributions_definitions = definitions["distributions"]
    for distribution, definition in distributions_definitions.items():

        if "parameters" in definition:
            dist_param_names = definition["parameters"]
        else:
            dist_param_names = None
        distributions[
            distribution
        ] = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
            name=distribution,
            distribution=definition["fun"],
            dist_param_names=dist_param_names,
        )

    params_definitions = definitions["parameters"]
    if "__comment" in params_definitions:
        del params_definitions["__comment"]

    for sectionlist, params in params_definitions.items():

        if sectionlist == "global":
            seclist_locs = None
            is_global = True
            is_dist = False
        elif "distribution_" in sectionlist:
            is_dist = True
            seclist_locs = None
            is_global = False
            dist_name = sectionlist.split("distribution_")[1]
            dist = distributions[dist_name]
        else:
            seclist_locs = multi_locations(sectionlist)
            is_global = False
            is_dist = False

        for param_config in params:
            param_name = param_config["name"]

            if isinstance(param_config["val"], (list, tuple)):
                is_frozen = False
                bounds = param_config["val"]
                value = None
            else:
                is_frozen = True
                value = param_config["val"]
                bounds = None

            if is_global:
                parameters.append(
                    ephys.parameters.NrnGlobalParameter(
                        name=param_name,
                        param_name=param_name,
                        frozen=is_frozen,
                        bounds=bounds,
                        value=value,
                    )
                )
            elif is_dist:
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

            else:

                if "dist" in param_config:
                    dist = distributions[param_config["dist"]]
                    use_range = True
                else:
                    dist = distributions["uniform"]
                    use_range = False

                if use_range:
                    parameters.append(
                        ephys.parameters.NrnRangeParameter(
                            name="%s.%s" % (param_name, sectionlist),
                            param_name=param_name,
                            value_scaler=dist,
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
                            value_scaler=dist,
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

    mechanisms = []
    for sectionlist, channels in mechanisms_definition.items():

        seclist_locs = multi_locations(sectionlist)

        for channel in channels["mech"]:
            mechanisms.append(
                ephys.mechanisms.NrnMODMechanism(
                    name="%s.%s" % (channel, sectionlist),
                    mod_path=None,
                    prefix=channel,
                    locations=seclist_locs,
                    preloaded=True,
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
                to morph_modifiers

    Returns:
        bluepyopt.ephys.morphologies.NrnFileMorphology: a morphology object
    """
    if morph_modifiers is None:
        morph_modifiers = [replace_axon_with_taper]
        morph_modifiers_hoc = [replace_axon_hoc]  # TODO: check the hoc is correct
        logger.debug("No morph_modifiers provided, we will use replace_axon_with_taper")

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
    morph_path,
    mechanisms,
    parameters,
    do_replace_axon=True,
    morph_modifiers=None,
    morph_modifiers_hoc=None,
):
    """Create a cell model based on a morphology, mechanisms and parameters

    Args:
        name (str): name of the model
        morph_path (str): path a morphology file
        mechanisms (dict): see docstring of function define_mechanisms for the
            format
        parameters (dict):  see docstring of function define_parameters for the
            format
        do_replace_axon (bool): replace axon with taper
        morph_modifiers (list): list of functions to modify morphologies
        morph_modifiers_hoc (list): list of hoc functions to modify morphologies

    Returns:
        CellModel
    """
    morph = define_morphology(
        str(morph_path),
        do_set_nseg=True,
        nseg_frequency=40,
        morph_modifiers=morph_modifiers,
        morph_modifiers_hoc=morph_modifiers_hoc,
    )

    mechs = define_mechanisms(mechanisms)
    params = define_parameters(parameters)

    return ephys.models.CellModel(
        name=name,
        morph=morph,
        mechs=mechs,
        params=params,
    )


def create_cell_models(
    emodel, working_dir, morphologies, mechanisms, parameters, morph_modifiers=None
):
    """Create cell models based on morphologies. The same mechanisms and
    parameters will be used for all morphologies

    Args:
        emodel (str): name of the e-model
        working_dir (str): path to the cwd
        morphologies (list): list of morphologies of the format
            morphologies = [{'name': morph_name, 'path': morph_path}]
        mechanisms (dict): see docstring of function define_mechanisms for the
            format
        parameters (dict):  see docstring of function define_parameters for the
            format
        morph_modifiers (list): list of morphology modifiers
    """
    cell_models = []
    for morphology in morphologies:

        morph_name = morphology["name"]
        morph_path = Path(morphology["path"])

        # Create the cell model
        name = "{}_{}".format(emodel, morph_name)
        cell_models.append(
            create_cell_model(
                name=name,
                morph_path=morph_path,
                mechanisms=mechanisms,
                parameters=parameters,
                morph_modifiers=morph_modifiers,
            )
        )

    return cell_models
