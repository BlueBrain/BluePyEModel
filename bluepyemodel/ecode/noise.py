"""Noise stimulus class"""
import logging

import numpy
import pkg_resources

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class Noise(BPEM_stimulus):

    """Noise current stimulus"""

    name = "Noise"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        self.holding_current = kwargs.get("holding_current", None)

        self.mu = kwargs.get("mu", None)
        self.data_filepath = kwargs.get("data_filepath", None)

        if data_filepath is None:
            raise Exception(
                "Please, set data_filepath or use NoiseOU3 or WhiteNoise stimuli"
                + "to get the default noise data files."
            )

        series = numpy.loadtxt(data_filepath)

        self.current_series = series[1]
        self.time_series = series[0]

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
            raise Exception("For eCode {}, dt has to be 0.1ms.".format(self.name))

        current = self.holding_current + self.current_series * (self.mu / 2.) + self.mu

        return self.time_series, current
