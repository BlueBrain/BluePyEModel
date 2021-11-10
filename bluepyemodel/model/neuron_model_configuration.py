"""Neuron Model Configuration"""
import logging
import pathlib

from bluepyemodel.model.distribution_configuration import DistributionConfiguration
from bluepyemodel.model.mechanism_configuration import MechanismConfiguration
from bluepyemodel.model.morphology_configuration import MorphologyConfiguration
from bluepyemodel.model.parameter_configuration import ParameterConfiguration

logger = logging.getLogger(__name__)

multiloc_map = {
    "all": ["apical", "basal", "somatic", "axonal"],
    "alldend": ["apical", "basal"],
    "somadend": ["apical", "basal", "somatic"],
    "somaxon": ["axonal", "somatic"],
    "allact": ["apical", "basal", "somatic", "axonal"],
}


class NeuronModelConfiguration:
    """A neuron model configuration, which includes the model's parameters, distributions,
    mechanisms and a morphology."""

    def __init__(self, configuration_name, available_mechanisms=None, available_morphologies=None):
        """Creates a model configuration, which includes the model parameters, distributions,
        mechanisms and a morphology.

        Args:
            configuration_name (str): name of the configuration, can be any string.
            available_mechanisms (list of str): list of the names of the available mechanisms in
                the "./mechanisms" directory for the local access point or on Nexus for the
                Nexus access point.
            available_morphologies (list of str): list of the names of the available morphology in
                the "./morphology" directory for the local access point or on Nexus for the
                Nexus access point.
        """

        self.configuration_name = configuration_name

        self.parameters = []
        self.mechanisms = []
        self.distributions = []
        self.morphology = None

        # TODO: actually use this:
        self.mapping_multilocation = None

        self.available_mechanisms = available_mechanisms
        if self.available_mechanisms is not None:
            self.available_mechanisms = set(self.available_mechanisms)
            self.available_mechanisms.add("pas")

        self.available_morphologies = available_morphologies

    @property
    def mechanism_names(self):
        """Returns the names of all the mechanisms used in the model"""

        return {m.name for m in self.mechanisms}

    @property
    def distribution_names(self):
        """Returns the names of all the distributions registered"""

        return {d.name for d in self.distributions}

    @property
    def used_distribution_names(self):
        """Returns the names of all the distributions used in the model"""

        return {p.distribution for p in self.parameters if p.distribution != "uniform"}

    @staticmethod
    def _format_locations(locations):
        """ """

        if locations is None:
            return []
        if isinstance(locations, str):
            return [locations]

        return locations

    def init_from_dict(self, configuration_dict):
        """Instantiate the object from its dictionary form"""

        self.configuration_name = configuration_dict["name"]

        if "distributions" in configuration_dict:
            for distribution in configuration_dict["distributions"]:
                self.add_distribution(
                    distribution["name"],
                    distribution["function"],
                    distribution.get("parameters", None),
                    distribution.get("soma_ref_location", None),
                )

        if "parameters" in configuration_dict:
            for param in configuration_dict["parameters"]:
                self.add_parameter(
                    param["name"],
                    param["location"],
                    param["value"],
                    param.get("mechanism", None),
                    param.get("dist", None),
                )

        if "mechanisms" in configuration_dict:
            for mechanism in configuration_dict["mechanisms"]:
                self.add_mechanism(
                    mechanism["name"], mechanism["location"], mechanism.get("stochastic", None)
                )

        self.morphology = MorphologyConfiguration(**configuration_dict["morphology"])

    def init_from_legacy_dict(self, parameters, morphology):
        """Instantiate the object from its legacy dictionary form"""

        ignore = ["v_init", "celsius", "cm", "Ra", "ena", "ek"]

        set_mechanisms = []
        for loc in parameters["mechanisms"]:
            set_mechanisms += parameters["mechanisms"][loc]["mech"]
        set_mechanisms = set(set_mechanisms)

        for distr_name, distr in parameters["distributions"].items():
            self.add_distribution(
                distr_name,
                distr["fun"],
                distr.get("parameters", None),
                distr.get("soma_ref_location", None),
            )

        for location in parameters["parameters"]:

            if location == "__comment":
                continue

            for param in parameters["parameters"][location]:

                mechanism = None

                if param["name"] not in ignore and "distribution" not in location:
                    mechanism = next((m for m in set_mechanisms if m in param["name"]), None)
                    if mechanism is None:
                        raise Exception(
                            f"Could not find mechanism associated to parameter {param['name']}"
                        )

                self.add_parameter(
                    param["name"],
                    location,
                    param["val"],
                    mechanism,
                    param.get("dist", None),
                )

        for location in parameters["mechanisms"]:
            for mech in parameters["mechanisms"][location]["mech"]:
                self.add_mechanism(mech, location)

        self.morphology = MorphologyConfiguration(**morphology)

    def add_distribution(self, distribution_name, function, parameters=None, soma_ref_location=0.5):
        """Add a channel distribution to the configuration

        Args:
            distribution_name (str): name of the distribution.
            function (str): python function of the distribution as a string. Will be executed
                using the python "eval" method.
            parameters (list of str): names of the parameters that parametrize the above function
                (no need to include the parameter "distance"). (Optional).
            soma_ref_location (float): location along the soma used as origin
                from which to compute the distances. Expressed as a fraction
                (between 0.0 and 1.0). (Optional).
        """

        tmp_distribution = DistributionConfiguration(
            name=distribution_name,
            function=function,
            parameters=parameters,
            soma_ref_location=soma_ref_location,
        )

        if any(tmp_distribution.name == d.name for d in self.distributions):
            raise Exception(f"Distribution {tmp_distribution.name} already exists")

        self.distributions.append(tmp_distribution)

    def add_parameter(
        self,
        parameter_name,
        locations,
        value,
        mechanism=None,
        distribution_name=None,
        stochastic=None,
    ):
        """Add a parameter to the configuration

        Args:
            parameter_name (str): name of the parameter. If related to a mechanisms, has to match
                the name of the parameter in the mod file.
            locations (str or list of str): sections of the neuron on which these parameters
                will be instantiated.
            value (float or list of two floats): if float, set the value of the parameter. If list
                of two floats, sets the upper and lower bound between which the parameter will
                be optimized.
            mechanism (name): name of the mechanism to which the parameter relates (optional).
            distribution_name (str): name of the distribution followed by the parameter (optional).
                Distributions have to be added before adding the parameters that uses them.
            stochastic (bool): Can the mechanisms to which the parameter relates behave
             stochastically (optional).
        """

        if not locations:
            raise Exception(
                "Cannot add a parameter without specifying a location. If "
                "global parameter, put 'global'."
            )

        locations = self._format_locations(locations)

        if distribution_name and distribution_name != "uniform":
            if distribution_name not in self.distribution_names:
                raise Exception(
                    f"No distribution of name {distribution_name} in the configuration."
                    " Please register your distributions first."
                )

        for loc in locations:

            tmp_param = ParameterConfiguration(
                name=parameter_name,
                location=loc,
                value=value,
                distribution=distribution_name,
                mechanism=mechanism,
            )

            if any(p == tmp_param for p in self.parameters):
                raise Exception(f"Parameter {parameter_name} is already at location {loc} or 'all'")

            self.parameters.append(tmp_param)

            if mechanism:
                self.add_mechanism(mechanism, loc, stochastic=stochastic)

    def is_mechanism_available(self, mechanism_name):
        """Is the mechanism part of the mechanisms available"""

        if self.available_mechanisms is not None:
            for mech in self.available_mechanisms:
                if mechanism_name in mech:
                    return True
            return False

        return True

    def add_mechanism(self, mechanism_name, locations, stochastic=None):
        """Add a mechanism to the configuration. This function should rarely be called directly as
         mechanisms are added automatically when using add_parameters. But it might be needed if a
         mechanism is not associated to any parameters.

        Args:
             mechanism_name (str): name of the mechanism.
             locations (str or list of str): sections of the neuron on which this mechanism
                 will be instantiated.
             stochastic (bool): Can the mechanisms behave stochastically (optional).
        """

        locations = self._format_locations(locations)

        for loc in locations:

            if not self.is_mechanism_available(mechanism_name):
                raise Exception(
                    f"You are trying to add mechanism {mechanism_name} but it is not available"
                    " on Nexus or local."
                )

            tmp_mechanism = MechanismConfiguration(
                name=mechanism_name, location=loc, stochastic=stochastic
            )

            # Check if mech is not already part of the configuration
            for m in self.mechanisms:
                if m.name == mechanism_name and m.location == loc:
                    return

            # Handle the case where the new mech is a key of the multilocation map
            if loc in multiloc_map:
                tmp_mechanisms = []
                for m in self.mechanisms:
                    if not (m.name == mechanism_name and m.location in multiloc_map[loc]):
                        tmp_mechanisms.append(m)
                self.mechanisms = tmp_mechanisms + [tmp_mechanism]

            # Handle the case where the new mech is a value of the multilocation map
            else:
                for m in self.mechanisms:
                    if m.name == tmp_mechanism.name:
                        if m.location in multiloc_map and loc in multiloc_map[m.location]:
                            break
                else:
                    self.mechanisms.append(tmp_mechanism)

    def select_morphology(
        self,
        morphology_name=None,
        morphology_path=None,
        morphology_format=None,
        seclist_names=None,
        secarray_names=None,
        section_index=None,
    ):
        """Select the morphology on which the neuron model will be based. Its name has to be part
        of the availabel morphologies.

        Args:
            morphology_name (str): name of the morphology. Optional if morphology_path is informed.
            morphology_path (str): path to the morphology file. If morphology_name is informed, has
                to match the stem of the morphology_path. Optional if morphology_name is informed.
            morphology_format (str): format of the morphology (asc or swc). If morphology_path is
                informed, has to match its suffix. Optional if morphology_format is informed.
            seclist_names (list): Names of the lists of sections (['somatic', ...]) (optional).
            secarray_names (list): names of the sections (['soma', ...]) (optional).
            section_index (int): index to a specific section, used for non-somatic
                recordings (optional).
        """

        if morphology_name not in self.available_morphologies:
            raise Exception(
                "The morphology you want to use is not available in Nexus or in your "
                "morphology directory"
            )

        self.morphology = MorphologyConfiguration(
            name=morphology_name,
            path=morphology_path,
            format=morphology_format,
            seclist_names=seclist_names,
            secarray_names=secarray_names,
            section_index=section_index,
        )

    def remove_parameter(self, parameter_name, locations=None):
        """Remove a parameter from the configuration. If locations is None or [], the whole
        parameter will be removed. WARNING: that does not remove automatically the mechanism
        which might be still use by other parameter"""

        locations = self._format_locations(locations)

        if locations:
            self.parameters = [
                p
                for p in self.parameters
                if p.name != parameter_name or p.location not in locations
            ]
        else:
            self.parameters = [p for p in self.parameters if p.name != parameter_name]

    def remove_mechanism(self, mechanism_name, locations=None):
        """Remove a mechanism from the configuration and all the associated parameters"""

        locations = self._format_locations(locations)

        if locations:
            self.mechanisms = [
                m
                for m in self.mechanisms
                if m.name != mechanism_name or m.location not in locations
            ]
            self.parameters = [
                p
                for p in self.parameters
                if p.mechanism != mechanism_name or p.location not in locations
            ]
        else:
            self.mechanisms = [m for m in self.mechanisms if m.name != mechanism_name]
            self.parameters = [p for p in self.parameters if p.mechanism != mechanism_name]

    def as_dict(self):
        """Returns the configuration as dict of parameters, mechanisms and
        a list of mechanism names"""

        return {
            "name": self.configuration_name,
            "mechanisms": [m.as_dict() for m in self.mechanisms],
            "distributions": [d.as_dict() for d in self.distributions],
            "parameters": [p.as_dict() for p in self.parameters],
            "morphology": self.morphology.as_dict(),
        }

    def __str__(self):
        """String representation"""

        str_form = f"Model Configuration - {self.configuration_name}:\n\n"

        str_form += "Mechanisms:\n"
        for m in self.mechanisms:
            str_form += f"   {m.as_dict()}\n"

        str_form += "Distributions:\n"
        for d in self.distributions:
            str_form += f"   {d.as_dict()}\n"

        str_form += "Parameters:\n"
        for p in self.parameters:
            str_form += f"   {p.as_dict()}\n"

        str_form += "Morphology:\n"
        str_form += f"   {self.morphology.as_dict()}\n"

        return str_form