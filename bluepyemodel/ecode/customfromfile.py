"""CustomFromFile stimulus class"""
import logging

import numpy

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class CustomFromFile(BPEM_stimulus):

    """CustomFromFile current stimulus to be loaded from a file"""

    name = "CustomFromFile"

    def __init__(self, location, **kwargs):
        """Constructor

        Args:
            location(Location): location of stimulus
            **kwargs: See below

        Keyword Arguments:
            data_filepath (str): path to the noise .txt data file. The file should have two columns:
                time (ms) and current (nA).
        """

        data_filepath = kwargs["data_filepath"]
        series = numpy.loadtxt(data_filepath)

        self.time_series = series[:, 0]
        self.current_series = series[:, 1]

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

        return self.time_series, self.current_series
