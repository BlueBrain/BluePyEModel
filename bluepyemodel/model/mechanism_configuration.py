"""Mechanism Configuration"""

from itertools import chain


class MechanismConfiguration:
    """Contains the information related to the definition and configuration of a mechanism"""

    def __init__(
        self,
        name,
        location,
        stochastic=None,
        version=None,
        parameters=None,
        ion_currents=None,
        nonspecific_currents=None,
        ionic_concentrations=None,
    ):
        """Init

        Args:
             name (str): name of the mechanism.
             locations (str or list of str): sections of the neuron on which this mechanism
                 will be instantiated.
             stochastic (bool): Can the mechanisms behave stochastically (optional).
             version (str): version id of the mod file.
             parameters (list): list of the possible parameter for this mechanism.
             ion_currents (list): list of the ion currents that this mechanism writes.
             nonspecific_currents (list): list of non-specific currents
             ionic_concentrations (list): list of the ionic concentration linked to the ion current
                If None, will be deduced from the ions list.
        """

        self.name = name
        self.location = location
        self.version = version
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
        }
