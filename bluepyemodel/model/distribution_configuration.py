"""Distribution Configuration"""


class DistributionConfiguration:
    """Contains all the information related to the definition and configuration of a parameter
    distribution"""

    def __init__(self, name, function=None, parameters=None, soma_ref_location=0.5):
        """Init

        Args:
            name (str): name of the distribution.
            function (str): python function of the distribution as a string. Will be executed
                using the python "eval" method.
            parameters (list of str): names of the parameters that parametrize the above function
                (no need to include the parameter "distance"). (Optional).
            soma_ref_location (float): location along the soma used as origin
                from which to compute the distances. Expressed as a fraction
                (between 0.0 and 1.0). (Optional).
        """

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
