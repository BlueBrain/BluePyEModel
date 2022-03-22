"""Mechanism Configuration"""


class MechanismConfiguration:
    """Contains the information related to the definition and configuration of a mechanism"""

    def __init__(self, name, location, stochastic=None, version=None, parameters=None, ions=None, ionic_concentrations=None):
        """Init

        Args:
             name (str): name of the mechanism.
             locations (str or list of str): sections of the neuron on which this mechanism
                 will be instantiated.
             stochastic (bool): Can the mechanisms behave stochastically (optional).
             version (str): version id of the mod file.
             parameters (list): list of the possible parameter for this mechanism.
             ions (list): list of the ion(s) that this mechanism writes.
             ionic_concentrations (list): list of the ionic concentration linked to the ion current.
                If None, will be deduced from the ions list.
        """

        self.name = name
        self.location = location
        self.version = version
        self.ions = ions
        self.ionic_concentrations = ionic_concentrations
        if self.ionic_concentrations is None:
            self.ionic_concentrations = []
            if self.ions is not None:
                for ion in self.ions:
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
