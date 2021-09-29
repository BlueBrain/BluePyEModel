"""Noise stimulus class"""
import logging

import numpy
import pkg_resources

from bluepyemodel.ecode.noise import NoiseMixin

logger = logging.getLogger(__name__)


class NoiseOU3(NoiseMixin):

    """NoiseOU3 current stimulus"""

    name = "NoiseOU3"

    def __init__(self, location, **kwargs):
        """Constructor

        Args:
            location(Location): location of stimulus
            **kwargs: See below

        Keyword Arguments:
            holding_current (float): amplitude of the holding current (nA)
            mu (float): mu is a factor in [nA] changing the noise from the file as
                Noise_{injected} = Noise_{file} * (mu/2) + mu
            data_filepath (str): path to the noise .txt data file
                If not given, will use the default one at bluepyemodel/ecodes/data/NoiseOU3.txt
        """

        self.holding_current = kwargs.get("holding_current", None)
        self.threshold_current = None

        self.mu = kwargs.get("mu", None)
        data_filepath = kwargs.get("data_filepath", None)

        if data_filepath is not None:
            series = numpy.loadtxt(data_filepath)
        else:
            series_file = pkg_resources.resource_filename(__name__, "data/NoiseOU3.txt")
            series = numpy.loadtxt(series_file)

        self.current_series = series[:, 1]
        self.time_series = series[:, 0]

        super().__init__(
            location=location,
        )
