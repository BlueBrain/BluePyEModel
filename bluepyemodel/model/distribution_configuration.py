"""Distribution Configuration"""

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


class DistributionConfiguration:
    """Contains all the information related to the definition and configuration of a parameter
    distribution"""

    def __init__(
        self,
        name,
        function=None,
        parameters=None,
        morphology_dependent_parameters=None,
        soma_ref_location=0.5,
        comment=None,
    ):
        """Init

        Args:
            name (str): name of the distribution.
            function (str): python function of the distribution as a string. Will be executed
                using the python "eval" method. The string needs to include "value" which will be
                replaced by the conductance of the parameter using the present distribution.
                It can also include "distance" if the distribution is parametrized by the distance
                to the soma. Example: "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}".
            parameters (list of str): names of the parameters that parametrize the above function
                if any. Note that "value" and "distance" do not need to be specified here.
            morphology_dependent_parameters (list of str): unused. To be deprecated.
            soma_ref_location (float): location along the soma used as origin from which to
                compute the distances. Expressed as a fraction (between 0.0 and 1.0).
            comment (str): additional comment or note.
        """

        self.name = name
        self.function = function

        if parameters is None:
            self.parameters = []
        elif isinstance(parameters, str):
            self.parameters = [parameters]
        else:
            self.parameters = parameters

        if morphology_dependent_parameters is None:
            self.morphology_dependent_parameters = []
        elif isinstance(morphology_dependent_parameters, str):
            self.morphology_dependent_parameters = [morphology_dependent_parameters]
        else:
            self.morphology_dependent_parameters = morphology_dependent_parameters

        if soma_ref_location is None:
            soma_ref_location = 0.5
        self.soma_ref_location = soma_ref_location

        self.comment = comment

    def as_dict(self):
        distr_dict = {
            "name": self.name,
            "function": self.function,
            "soma_ref_location": self.soma_ref_location,
        }

        for attr in ["parameters", "morphology_dependent_parameters", "comment"]:
            if getattr(self, attr):
                distr_dict[attr] = getattr(self, attr)

        return distr_dict

    def as_legacy_dict(self):
        distr_dict = {"fun": self.function}

        if self.parameters:
            distr_dict["parameters"] = self.parameters

        return distr_dict
