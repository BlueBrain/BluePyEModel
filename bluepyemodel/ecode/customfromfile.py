"""CustomFromFile stimulus class"""

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
