"""Ramp stimulus class"""

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


class Ramp(BPEM_stimulus):
    """Ramp current stimulus

    .. code-block:: none

            holdi          holdi+amp       holdi
              :                :             :
              :                :             :
              :               /|             :
              :              / |             :
              :             /  |             :
              :            /   |             :
              :           /    |             :
              :          /     |             :
              :         /      |             :
              :        /       |             :
              :       /        |             :
        |___________ /         |__________________________
        ^           ^          ^                          ^
        :           :          :                          :
        :           :          :                          :
        t=0         delay      delay+duration             totduration
    """

    name = "Ramp"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        self.amp = kwargs.get("amp", None)
        self.amp_rel = kwargs.get("thresh_perc", 200.0)

        self.holding_current = kwargs.get("holding_current", None)
        self.threshold_current = None

        if self.amp is None and self.amp_rel is None:
            raise TypeError(f"In stimulus {self.name}, amp and thresh_perc cannot be both None.")

        self.delay = kwargs.get("delay", 250.0)
        self.duration = kwargs.get("duration", 1350.0)
        self.total_duration = kwargs.get("totduration", 1850.0)

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.delay

    @property
    def stim_end(self):
        return self.delay + self.duration

    @property
    def amplitude(self):
        if self.amp_rel is None or self.threshold_current is None:
            return self.amp
        return self.threshold_current * (float(self.amp_rel) / 100.0)

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        holding_current = self.holding_current if self.holding_current is not None else 0

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()

        self.time_vec.append(0.0)
        self.current_vec.append(holding_current)

        self.time_vec.append(self.delay)
        self.current_vec.append(holding_current)

        self.time_vec.append(self.stim_end)
        self.current_vec.append(holding_current + self.amplitude)

        self.time_vec.append(self.stim_end)
        self.current_vec.append(holding_current)

        self.time_vec.append(self.total_duration)
        self.current_vec.append(holding_current)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )

    def generate(self, dt=0.1):
        """Return current time series"""
        holding_current = self.holding_current if self.holding_current is not None else 0

        t = numpy.arange(0.0, self.total_duration, dt)
        current = numpy.full(t.shape, holding_current, dtype="float64")

        ton_idx = int(self.stim_start / dt)
        toff_idx = int((self.stim_end) / dt)

        current[ton_idx:toff_idx] += numpy.linspace(0.0, self.amplitude, toff_idx - ton_idx + 1)[
            :-1
        ]

        return t, current
