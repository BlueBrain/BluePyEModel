"""Parameter Configuration"""


class ParameterConfiguration:
    """"""

    def __init__(self, name, location, value, distribution="uniform", mechanism=None):

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
