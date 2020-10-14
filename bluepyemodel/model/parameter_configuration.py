"""Parameter Configuration"""

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
                be optimised.
            distribution (str): name of the distribution followed by the parameter (optional).
            mechanism (name): name of the mechanism to which the parameter relates (optional).
        """

        self.name = name
        self.location = location

        self.value = value
        if isinstance(self.value, tuple):
            self.value = list(self.value)
        if isinstance(self.value, list) and len(self.value) == 1:
            self.value = self.value[0]

        self.mechanism = mechanism

        self.distribution = distribution
        if self.distribution is None:
            self.distribution = "uniform"

    @property
    def valid_value(self):
        return not (self.value is None)

    def as_dict(self):
        """ """

        param_dict = {
            "name": self.name,
            "value": self.value,
            "location": self.location,
        }

        if self.distribution and self.distribution != "uniform":
            param_dict["distribution"] = self.distribution

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
