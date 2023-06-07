"""Comb stimulus class."""

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


class Comb(BPEM_stimulus):
    # pylint: disable=line-too-long,anomalous-backslash-in-string

    """Comb current stimulus which consists of regularly spaced short steps, aimed at
    generating a train of spikes.

    .. code-block:: none

              holdi         amp       holdi        amp       holdi           .   .   .
                :            :          :           :          :
                :       ___________     :      ___________     :     ___                  _____
                :      |           |    :     |           |    :    |                          |
                :      |           |    :     |           |    :    |        * n_steps         |
                :      |           |    :     |           |    :    |        .   .   .         |
                :      |           |    :     |           |    :    |                          |
        |______________|           |__________|           |_________|                          |_____
        :              :           :          :           :         :                                ^
        :              :           :          :           :         :                                :
        :              :           :          :           :         :                                :
         <--  delay  --><-duration->           <-duration->         :        .   .   .     totduration
                        <--   inter_delay  --><--  inter_delay   -->

    """

    name = "Comb"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
            inter_delay (float): time between each step beginnings in ms
            n_steps (int): number of steps for the stimulus
            amp (float): amplitude of each step(nA)
            delay (float): time at which the first current spike begins (ms)
            duration (float): duration of each step (ms)
            totduration (float): total duration of the whole stimulus (ms)
        """
        self.inter_delay = kwargs.get("inter_delay", 5)
        self.n_steps = kwargs.get("n_steps", 20)
        self.amp = kwargs.get("amp", 40)
        self.delay = kwargs.get("delay", 200.0)
        self.duration = kwargs.get("duration", 0.5)
        self.total_duration = kwargs.get("totduration", 350.0)
        self.holding = 0.0  # hardcoded holding for now (holding_current is modified externally)

        if self.stim_end > self.total_duration:
            raise ValueError(
                "stim_end is larger than total_duration: {self.stim_end} > {self.total_duration})"
            )

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.delay

    @property
    def stim_end(self):
        return self.delay + self.n_steps * self.duration

    @property
    def amplitude(self):
        return self.amp

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()
        self.time_vec.append(self.holding)
        self.current_vec.append(self.holding)

        for step in range(self.n_steps):
            _delay = step * self.inter_delay + self.delay
            self.time_vec.append(_delay)
            self.current_vec.append(self.holding)

            self.time_vec.append(_delay)
            self.current_vec.append(self.amplitude)

            self.time_vec.append(_delay + self.duration)
            self.current_vec.append(self.amplitude)

            self.time_vec.append(_delay + self.duration)
            self.current_vec.append(self.holding)

        self.time_vec.append(self.total_duration)
        self.current_vec.append(self.holding)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )

    def generate(self, dt=0.1):
        """Return current time series"""

        t = numpy.arange(0.0, self.total_duration, dt)
        current = numpy.full(t.shape, self.holding, dtype="float64")

        for step in range(self.n_steps):
            _delay = step * self.inter_delay + self.delay
            ton_idx = int(_delay / dt)
            toff_idx = int((_delay + self.duration) / dt)
            current[ton_idx:toff_idx] = self.amplitude

        return t, current
