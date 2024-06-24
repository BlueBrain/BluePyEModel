"""Methods to handle cell model configuration."""

"""
Copyright 2023-2024, EPFL/Blue Brain Project

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


from dataclasses import asdict
from dataclasses import dataclass


@dataclass
class Parameter:
    """Used to exchange configuration information with BPEM."""

    name: str  # Neuron parameter name, i.e. {param}_{suffix}
    value: tuple  # Optimization boundaries
    location: str  # Cell compartment
    mechanism: str = ""  # Mechanism suffix
    distribution: str = ""  # Morphology dependent function


class Configuration:
    """Class to collect cell model parameters and generate a
    configuration."""

    def __init__(self):
        self._parameters = []

    @staticmethod
    def _clean_up_list_of_dicts(list_of_dicts, sort_by=None):
        """Sort list of dicts and remove duplicates.

        Args:
            sort_by (str): dict key to sort by
        """

        # Make unique list
        list_of_dicts = [dict(s) for s in set(frozenset(x.items()) for x in list_of_dicts)]
        # Sort by key
        if sort_by:
            list_of_dicts = sorted(list_of_dicts, key=lambda k: k[sort_by])
        return list_of_dicts

    def _add_mech_param(self, mech, param):
        """Add parameter from provided mechanism.

        Args:
            mech (Mechanism): mechanism containing parameter info
            param (str): name of parameter to add
        """

        model = mech.model
        suffix = model["suffix"]
        locations = asdict(mech.distribution)
        name = param + "_" + suffix
        for comp, distr in locations.items():
            if not distr == "":
                pset = Parameter(
                    name=name,
                    value=tuple(mech.get_bounds(param)),
                    location=comp,
                    mechanism=suffix,
                    distribution=distr,
                )
                self._parameters.append(pset)

    def add_from_mechanism(self, mech):
        """Add all parameters from provided mechanism.

        Args:
            mech (Mechanism): mechanism to add
        """

        model = mech.model

        if "gbar" in model:
            param = model["gbar"]
            self._add_mech_param(mech, param)

        if "parameters" in model:
            for param in model["parameters"]:
                self._add_mech_param(mech, param)

    def add_parameter(self, name, location, value):
        """Manually add a cell model parameter.

        Args:
            name (str): parameter name
            location (str): compartment to insert parameter
            value (tuple): bounds for parameter optimization
        """

        pset = Parameter(name=name, location=location, value=value)
        self._parameters.append(pset)

    def get_mechanisms(self):
        """Returns a list of mechanisms and compartments to insert them."""

        mechanisms = [
            {"name": p.mechanism, "location": p.location} for p in self._parameters if p.mechanism
        ]
        return self._clean_up_list_of_dicts(mechanisms, "name")

    def get_parameters(self):
        """Returns a list of configuration parameters."""

        params = [asdict(p) for p in self._parameters]
        return self._clean_up_list_of_dicts(params, "mechanism")

    def get_distributions(self):
        """Returns a list of unique distribution definitions used."""

        distr = [{"name": p.distribution} for p in self._parameters if p.distribution]
        return self._clean_up_list_of_dicts(distr, "name")

    def __str__(self):
        locations = {}
        for param in self._parameters:
            if param.location not in locations:
                locations[param.location] = []
            locations[param.location].append([param.name, param.value, param.distribution])
        out_str = ["\n"]
        for loc, content in locations.items():
            out_str.append(f">>> {loc} <<<")
            for p in content:
                if p[2]:
                    out_str.append(f"    {p[0]}, value: {p[1]}, distribution: {p[2]}")
                else:
                    out_str.append(f"    {p[0]}, value: {p[1]}")
        return "\n".join(out_str)
