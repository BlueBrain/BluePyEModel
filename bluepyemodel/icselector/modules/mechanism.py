"""Mechanism class corresponding to mechanisms fields in the icmapping file."""

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


from pprint import pprint

import numpy as np

from .distribution import Distribution
from .distribution import asdict


class Mechanism:
    """Holds all collected information associated with a Neuron mechanism.
    Corresponds to mechanisms fields in the icmapping file, with added
    fields containing additional information from GeneSelector."""

    def __init__(self, **kwargs):
        self.nexus = {}
        self.model = {}
        self.requires = []
        self.status = None
        self.distribution = Distribution()
        self._mapped_from = []
        self._bounds = {}
        self._selected = False
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Some initial values
        if "parameters" in self.model:
            for param in self.model["parameters"]:
                self.set_parameters(**{param: 0})
        self.set_gbar([0, 1])

    def select(self, check_status=None):
        """Select mechanism for inclusion in model configuration.
        Args:
            check_status (str): only select the mechanism if status meets criterium
        """

        if not self.status:
            self._selected = True
            return

        if check_status == "stable":
            self._selected = check_status == self.status
        else:
            self._selected = True

    def deselect(self):
        """Deselect mechanism."""

        self._selected = False

    def is_selected(self):
        """Mechanism has been selected for inclusion in model configuration."""

        return self._selected

    def get_bounds(self, param):
        """Get parameter bounds.
        Args:
            param (str): parameter name

        Returns:
            bounds (list): parameter bounds
        """

        return self._bounds[param]

    def set_distribution(self, *args, **kwargs):
        """Set gbar distributions for all compartments.
        Args:
            args (list): distribution (str or Distribution)
            kwargs (dict): fields to be set specified as {compartment: distribution}
        """

        if len(args) > 0:
            if isinstance(args[0], Distribution):
                kwargs = asdict(args[0])
                self.distribution.set_fields(**kwargs)
            else:
                self.distribution.set_all(args[0])
        else:
            self.distribution.set_fields(**kwargs)

    def set_parameters(self, **kwargs):
        """Set the values of model parameters other than gbar
        Args:
            kwargs (dict): parameter values formatted as {parameter: value}
        """

        if "parameters" in self.model:
            for name, value in kwargs.items():
                if name in self.model["parameters"]:
                    if not isinstance(value, list):
                        value = [value]
                    self._bounds[name] = value
                else:
                    raise KeyError(f"Model '{self.model['suffix']}' has no parameter '{name}'")

    def set_gbar(self, value):
        """Set the value of gbar
        Args:
            value (list or number): value or bounds of gbar
        """

        if "gbar" in self.model:
            gbar = self.model["gbar"]
            if not isinstance(value, list):
                value = [value]
            value_check = [isinstance(v, (float, int)) for v in value]
            if not np.sum(value_check) == len(value):
                raise TypeError("gbar value must be numeric.")
            self._bounds[gbar] = value

    def set_from_gene_info(self, info):
        """Set fields based on gene info.
        Args:
            info (dict): channel information coming from the GeneSelector
        """

        self.set_distribution(info["distribution"])
        gbar = [0, info["gbar_max"]]
        self.set_gbar(gbar)
        self._mapped_from.append(f"{info['channel']}")

    def set_from_icmap(self, info):
        """Set fields based on icmap info.
        Args:
            info (dict): channel information coming from the icmapping file
        """

        if isinstance(info, list):
            self.set_gbar(info)
        elif isinstance(info, dict):
            for key, value in info.items():
                if key == "gbar":
                    self.set_gbar(value)
                elif key == "distribution":
                    if isinstance(value, str):
                        self.set_distribution(value)
                    elif isinstance(value, dict):
                        self.set_distribution(**value)
                elif key == "bounds":
                    self.set_parameters(**value)

    def asdict(self):
        """Return fields as a dict"""

        return self.__dict__

    def print(self):
        """Alternative print method"""

        pprint(self.asdict())

    def __str__(self):
        pstr = ", ".join([f"{k} = {v}" for k, v in self._bounds.items()])
        out_str = "{name}, distribution: {dist}, bounds: {param}"
        return out_str.format(name=self.model["suffix"], dist=self.distribution, param=pstr)
