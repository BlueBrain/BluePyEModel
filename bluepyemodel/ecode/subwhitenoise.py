"""SubWhiteNoise stimulus class"""
import logging

import numpy
import pkg_resources

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class SubWhiteNoise(BPEM_stimulus):

    """SubWhiteNoise current stimulus"""

    name = "SubWhiteNoise"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        self.amp = kwargs.get("amp", None)
        self.amp_rel = kwargs.get("thresh_perc", 150.0)

        self.holding_current = kwargs.get("holding_current", None)
        self.threshold_current = None

        if self.amp is None and self.amp_rel is None:
            raise Exception(f"In stimulus {self.name}, amp and thresh_perc cannot be both None.")

        series_file = pkg_resources.resource_filename(__name__, "data/subwhitenoise.npy")
        series = numpy.load(series_file)

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

    @property
    def amplitude(self):
        if self.amp_rel is None or self.threshold_current is None:
            return self.amp
        return self.threshold_current * (float(self.amp_rel) / 100.0)

    def generate(self, dt=0.1):
        """Return current time series"""

        if dt != 0.1:
            raise Exception(f"For eCode {self.name}, dt has to be 0.1ms.")

        current = self.holding_current + self.amplitude * self.current_series
        return self.time_series, current
