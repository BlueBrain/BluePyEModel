"""SineSpec stimulus class"""

"""
Copyright 2023-2024 Blue Brain Project / EPFL

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
            raise TypeError("In stimulus {self.name}, amp and thresh_perc cannot be both None.")

        self.delay = kwargs.get("delay", 0.0)
        self.duration = kwargs.get("duration", 5000.0)
        self.total_duration = kwargs.get("totduration", 5000.0)

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
    def amplitude(self):
        if self.amp_rel is None or self.threshold_current is None:
            return self.amp
        return self.threshold_current * (float(self.amp_rel) / 100.0)

    def generate(self, dt=0.1):
        """Return current time series"""
        holding_current = self.holding_current if self.holding_current is not None else 0

        t = numpy.arange(0.0, self.total_duration, dt)
        current = numpy.full(t.shape, holding_current, dtype="float64")

        ton_idx = int(self.stim_start / dt)
        toff_idx = int(self.stim_end / dt)

        t_sine = numpy.linspace(0.0, self.duration / 1e3, toff_idx - ton_idx + 1)[:-1]
        current_sine = self.amplitude * numpy.sin(
            2.0 * numpy.pi * (1.0 + (1.0 / (5.15 - (t_sine - 0.1)))) * (t_sine - 0.1)
        )

        current[ton_idx:toff_idx] += current_sine

        return t, current
