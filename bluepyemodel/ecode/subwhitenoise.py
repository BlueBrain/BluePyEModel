"""SubWhiteNoise stimulus class"""

"""
Copyright 2023, EPFL/Blue Brain Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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
            **kwargs: See below

        Keyword Arguments:
            amp (float): amplitude (nA) that multiplies the noise from the data file
                when the relative amplitude is not used
            thresh_perc (float): amplitude relative to the threshold current (%)
                that multiplies the noise from the data file
            holding_current (float): amplitude of the holding current (nA)
            data_filepath (str): path to the noise .txt data file
                If not given, will use the default one at bluepyemodel/ecodes/data/SubWhiteNoise.txt
        """

        self.amp = kwargs.get("amp", None)
        self.amp_rel = kwargs.get("thresh_perc", 150.0)

        self.holding_current = kwargs.get("holding_current", None)
        self.threshold_current = None

        if self.amp is None and self.amp_rel is None:
            raise TypeError(f"In stimulus {self.name}, amp and thresh_perc cannot be both None.")

        data_filepath = kwargs.get("data_filepath", None)

        if data_filepath is not None:
            series = numpy.loadtxt(data_filepath)
        else:
            series_file = pkg_resources.resource_filename(__name__, "data/SubWhiteNoise.txt")
            series = numpy.loadtxt(series_file)

        self.current_series = series[:, 1]
        self.time_series = series[:, 0]

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
        holding_current = self.holding_current if self.holding_current is not None else 0

        if dt != 0.1:
            raise ValueError(f"For eCode {self.name}, dt has to be 0.1ms.")

        current = holding_current + self.amplitude * self.current_series
        return self.time_series, current
