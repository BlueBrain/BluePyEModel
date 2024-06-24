"""Mechanism Configuration"""

"""
Copyright 2023-2024 Blue Brain Project / EPFL

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

from itertools import chain


class MechanismConfiguration:
    """Contains the information related to the definition and configuration of a mechanism"""

    def __init__(
        self,
        name,
        location,
        stochastic=None,
        version=None,
        temperature=None,
        ljp_corrected=None,
        parameters=None,
        ion_currents=None,
        nonspecific_currents=None,
        ionic_concentrations=None,
        id=None,
    ):
        """Init

        Args:
             name (str): name of the mechanism. Should be the SUFFIX from the mod file.
             locations (str or list of str): sections of the neuron on which this mechanism
                 will be instantiated.
             stochastic (bool): Can the mechanisms behave stochastically (optional).
             version (str): version id of the mod file.
             temperature (int): temperature of the mechanism
             ljp_corrected (bool): whether the mechanism is ljp corrected
             parameters (list): list of the possible parameter for this mechanism.
             ion_currents (list): list of the ion currents that this mechanism writes.
             nonspecific_currents (list): list of non-specific currents
             ionic_concentrations (list): list of the ionic concentration linked to the ion current
                If None, will be deduced from the ions list.
             id (str): Optional. Nexus ID of the mechanism.
        """

        self.name = name
        self.location = location
        self.version = version
        self.temperature = temperature
        self.ljp_corrected = ljp_corrected
        self.ion_currents = ion_currents
        self.nonspecific_currents = nonspecific_currents
        self.ionic_concentrations = ionic_concentrations
        if self.ionic_concentrations is None:
            self.ionic_concentrations = []
            if self.ion_currents is not None:
                for ion in self.ion_currents:
                    # remove 'i' in the front and put 'i' at the back to make it a concentration
                    self.ionic_concentrations.append(f"{ion[1:]}i")

        self.stochastic = stochastic
        if self.stochastic is None:
            self.stochastic = "Stoch" in self.name

        if parameters is None:
            self.parameters = {}
        elif isinstance(parameters, str):
            self.parameters = {parameters: [None, None]}
        else:
            self.parameters = parameters

        self.id = id

    def get_current(self):
        """Return the ion current names."""
        current = []
        ion_currents = self.ion_currents if self.ion_currents is not None else []
        nonspecific_currents = (
            self.nonspecific_currents if self.nonspecific_currents is not None else []
        )
        for curr in list(chain.from_iterable((ion_currents, nonspecific_currents))):
            current.append(f"{curr}_{self.name}")
        return current

    def as_dict(self):
        return {
            "name": self.name,
            "stochastic": self.stochastic,
            "location": self.location,
            "version": self.version,
            "temperature": self.temperature,
            "ljp_corrected": self.ljp_corrected,
            "id": self.id,
        }
