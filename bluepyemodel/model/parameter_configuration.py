"""Parameter Configuration"""


class ParameterConfiguration:
    """Contains all the information related to the definition and configuration of a parameter"""

    def __init__(self, name, location, value, distribution="uniform", mechanism=None):
        """Init

        Args:
            name (str): name of the parameter. If related to a mechanisms, has to match
                the name of the parameter in the mod file.
            location (str): section of the neuron on which the parameter will be instantiated.
            value (float or list of two floats): if float, set the value of the parameter. If list
                of two floats, sets the upper and lower bound between which the parameter will
                be optimized.
            distribution (str): name of the distribution followed by the parameter (optional).
            mechanism (name): name of the mechanism to which the parameter relates (optional).
        """

        self.name = name
        self.location = location
        self.value = value
        if isinstance(self.value, tuple):
            self.value = list(self.value)
        self.mechanism = mechanism

        self.distribution = distribution
        if self.distribution is None:
            self.distribution = "uniform"

    def as_dict(self):
        """ """

        param_dict = {
            "name": self.name,
            "value": self.value,
            "location": self.location,
        }

        if self.distribution and self.distribution != "uniform":
            param_dict["dist"] = self.distribution

        if self.mechanism:
            param_dict["mechanism"] = self.mechanism

        return param_dict

    def as_legacy_dict(self):
        """ """

        param_dict = {"name": self.name, "val": self.value}

        if self.distribution and self.distribution != "uniform":
            param_dict["dist"] = self.distribution

        return param_dict

    def __eq__(self, other):

        return self.name == other.name and (
            self.name == "all" or other.location == "all" or self.location == other.location
        )
