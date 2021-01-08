"""SineSpec stimulus class"""
import logging
import numpy

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class SineSpec(BPEM_stimulus):

    """SineSpec current stimulus"""

    name = "SineSpec"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        self.amp = kwargs.get("amp", None)
        self.amp_rel = kwargs.get("thresh_perc", 60.0)

        self.holding_current = kwargs.get("holding_current", None)
        self.threshold_current = None

        if self.amp is None and self.amp_rel is None:
            raise Exception("In stimulus %s, amp and thresh_perc cannot be both None." % self.name)

        self.delay = kwargs.get("delay", 0.0)
        self.duration = kwargs.get("duration", 5000.0)

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.delay

    @property
    def stim_end(self):
        return self.duration + self.delay

    @property
    def total_duration(self):
        return self.duration + self.delay

    @property
    def amplitude(self):
        if self.amp_rel is None or self.threshold_current is None:
            return self.amp
        return self.threshold_current * (float(self.amp_rel) / 100.0)

    def generate(self, dt=0.1):
        """Return current time series"""

        t = numpy.arange(0.0, self.total_duration, dt)
        current = numpy.full(t.shape, self.holding_current, dtype="float64")

        t_sine = numpy.arange(0.0, self.duration / 1e3, dt / 1e3)
        current_sine = self.amplitude * numpy.sin(
            2.0 * numpy.pi * (1.0 + (1.0 / (5.15 - (t_sine - 0.1)))) * (t_sine - 0.1)
        )

        ton_idx = int(self.stim_start / dt)
        toff_idx = int(self.stim_end / dt)
        current[ton_idx:toff_idx] += current_sine

        return t, current
