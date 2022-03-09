"""Mechanism Configuration"""


class MechanismConfiguration:
    """Contains the information related to the definition and configuration of a mechanism"""

    def __init__(self, name, location, stochastic=None, version=None, parameters=None, ions=None):
        """Init

        Args:
             name (str): name of the mechanism.
             locations (str or list of str): sections of the neuron on which this mechanism
                 will be instantiated.
             stochastic (bool): Can the mechanisms behave stochastically (optional).
             version (str): version id of the mod file.
             parameters (list): list of the possible parameter for this mechanism.
             ions (list): list of the ion(s) that this mechanism writes.
        """

        self.name = name
        self.location = location
        self.version = version
        self.ions = ions

        self.stochastic = stochastic
        if self.stochastic is None:
            self.stochastic = "Stoch" in self.name

        if parameters is None:
            self.parameters = {}
        elif isinstance(parameters, str):
            self.parameters = {parameters: [None, None]}
        else:
            self.parameters = parameters

    def get_ion_current(self):
        """Return the ion current names."""
        ion_current = []
        for ion in self.ions:
            ion_current.append(f"{ion}_{self.name}")
        return ion_current

    def as_dict(self):

        return {
            "name": self.name,
            "stochastic": self.stochastic,
            "location": self.location,
            "version": self.version,
        }
