"""Mechanism Configuration"""


class MechanismConfiguration:
    """"""

    def __init__(self, name, location, stochastic=None):

        self.name = name
        self.location = location
        self.stochastic = stochastic
        if self.stochastic is None:
            self.stochastic = "Stoch" in self.name

    def as_dict(self):

        return {"name": self.name, "stochastic": self.stochastic, "location": self.location}
