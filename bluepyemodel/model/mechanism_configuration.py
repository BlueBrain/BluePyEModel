"""Mechanism Configuration"""


class MechanismConfiguration:
    """Contains the information related to the definition and configuration of a mechanism"""

    def __init__(self, name, location, stochastic=None, version=None):
        """Init

        Args:
             name (str): name of the mechanism.
             locations (str or list of str): sections of the neuron on which this mechanism
                 will be instantiated.
             stochastic (bool): Can the mechanisms behave stochastically (optional).
             version (str): version id of the mod file.
        """

        self.name = name
        self.location = location
        self.version = version

        self.stochastic = stochastic
        if self.stochastic is None:
            self.stochastic = "Stoch" in self.name

    def as_dict(self):

        return {
            "name": self.name,
            "stochastic": self.stochastic,
            "location": self.location,
            "version": self.version,
        }
