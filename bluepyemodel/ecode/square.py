"""BPEM_stimulus class"""

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
from bluepyopt.ephys.stimuli import NrnSquarePulse

logger = logging.getLogger(__name__)


class BPOSquarePulse(NrnSquarePulse):
    """Abstract current stimulus based on BluePyOpt square stimulus.

    Can be used to reproduce results using BluePyOpt's NrnSquarePulse.

    .. code-block:: none

              holdi               holdi+amp                holdi
                :                     :                      :
                :                     :                      :
                :           ______________________           :
                :          |                      |          :
                :          |                      |          :
                :          |                      |          :
                :          |                      |          :
        |__________________|                      |______________________
        ^                  ^                      ^                      ^
        :                  :                      :                      :
        :                  :                      :                      :
        t=0                delay                  delay+duration         totduration
    """

    name = ""

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        super().__init__(
            step_amplitude=kwargs.get("amp", None),
            step_delay=kwargs.get("delay", 250.0),
            step_duration=kwargs.get("duration", 1350.0),
            total_duration=kwargs.get("totduration", 1850.0),
            location=location,
        )

        self.holding_current = kwargs.get("holding_current", None)
        self.holding_iclamp = None

    @property
    def stim_start(self):
        return self.step_delay

    @property
    def stim_end(self):
        return self.step_delay + self.step_duration

    @property
    def amplitude(self):
        return self.step_amplitude

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.step_duration

        self.iclamp.delay = self.step_delay
        self.iclamp.amp = self.step_amplitude

        if self.holding_current is not None:
            self.holding_iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
            self.holding_iclamp.dur = self.total_duration
            self.holding_iclamp.delay = 0
            self.holding_iclamp.amp = self.holding_current

    def destroy(self, sim=None):  # pylint:disable=W0613
        """Destroy stimulus"""
        self.iclamp = None
        self.holding_iclamp = None

    def generate(self, dt=0.1):
        """Return current time series"""
        holding_current = self.holding_current if self.holding_current is not None else 0

        t = numpy.arange(0.0, self.total_duration, dt)
        current = numpy.full(t.shape, holding_current, dtype="float64")

        ton_idx = int(self.stim_start / dt)
        toff_idx = int(self.stim_end / dt)

        current[ton_idx:toff_idx] += self.amplitude

        return t, current

    def __str__(self):
        """String representation"""
        return f"{self.name} current played at {self.location}"
