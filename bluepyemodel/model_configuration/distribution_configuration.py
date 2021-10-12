"""Distribution Configuration"""


class DistributionConfiguration:
    def __init__(self, name, function=None, parameters=None, soma_ref_location=0.5):

        self.name = name
        self.function = function

        self.parameters = parameters
        if self.parameters is None:
            self.parameters = []

        if soma_ref_location is None:
            soma_ref_location = 0.5
        self.soma_ref_location = soma_ref_location

    def as_dict(self):

        distr_dict = {
            "name": self.name,
            "function": self.function,
            "soma_ref_location": self.soma_ref_location,
        }

        if self.parameters:
            distr_dict["parameters"] = self.parameters

        return distr_dict

    def as_legacy_dict(self):

        distr_dict = {"fun": self.function}

        if self.parameters:
            distr_dict["parameters"] = self.parameters

        return distr_dict
