"""Noise stimulus class"""
import logging

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class NoiseMixin(BPEM_stimulus):

    """Noise current stimulus"""

    name = "Noise"

    def __init__(self, location):
        """Constructor

        Args:
            location(Location): location of stimulus
        """
        super().__init__(
            location=location,
        )

    @property
    def total_duration(self):
        return self.time_series[-1]

    @property
    def stim_start(self):
        return 0.0

    @property
    def stim_end(self):
        return self.time_series[-1]

    def generate(self, dt=0.1):
        """Return current time series"""

        if dt != 0.1:
            raise Exception(f"For eCode {self.name}, dt has to be 0.1ms.")

        current = self.holding_current + self.current_series * (self.mu / 2.0) + self.mu

        return self.time_series, current
