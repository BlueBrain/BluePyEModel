"""Noise stimulus class"""
import logging

import numpy
import pkg_resources

from bluepyemodel.ecode.noise import NoiseMixin

logger = logging.getLogger(__name__)


class WhiteNoise(NoiseMixin):

    """WhiteNoise current stimulus"""

    name = "WhiteNoise"

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
            raise Exception("In stimulus %s, amp and thresh_perc cannot be both None." % self.name)

        self.mu = kwargs.get("mu", None)
        data_filepath = kwargs.get("data_filepath", None)

        if data_filepath is not None:
            series = numpy.loadtxt(data_filepath)
        else:
            series_file = pkg_resources.resource_filename(__name__, "data/WhiteNoise.txt")
            series = numpy.loadtxt(series_file)

        self.current_series = series[:,1]
        self.time_series = series[:,0]

        super().__init__(
            location=location,
        )
