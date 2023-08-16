"""DeHyperpol stimulus class"""

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


class DeHyperpol(BPEM_stimulus):

    """DeHyperpol current stimulus

    .. code-block:: none

             holdi         holdi+amp        holdi+amp2     holdi
               :                :               :            :
               :         _________________      :            :
               :        |                 |     :            :
               :        |                 |     :            :
        |_______________|                 |     :       ___________
        ^               ^                 |     :      |           ^
        :               :                 |     :      |           :
        :               :                 |     :      |           :
        :               :                 |____________|           :
        :               :                 ^            ^           :
        :               :                 :            :           :
        :               :                 :            :           :
        t=0             delay             tmid         toff        totduration
    """

    name = "DeHyperpol"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        self.depol_amp = kwargs.get("amp", None)
        self.depol_amp_rel = kwargs.get("amp_rel", None)
        self.hyper_amp = kwargs.get("amp2", None)
        self.hyper_amp_rel = kwargs.get("amp2_rel", None)

        if self.hyper_amp is None and self.hyper_amp_rel is None:
            raise TypeError(
                f"In stimulus {self.name}, hyper_amp and hyper_amp_rel cannot be both None."
            )

        if self.depol_amp is None and self.depol_amp_rel is None:
            raise TypeError(
                f"In stimulus {self.name}, depol_amp and depol_amp_rel cannot be both None."
            )

        self.holding_current = kwargs.get("holding_current", None)
        self.threshold_current = None

        self.delay = kwargs.get("delay", 250.0)
        self.tmid = kwargs.get("tmid", 520.0)
        self.toff = kwargs.get("toff", 970.0)
        self.total_duration = kwargs.get("totduration", 1220.0)

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.delay

    @property
    def stim_end(self):
        return self.toff

    @property
    def amplitude(self):
        if self.hyper_amp_rel is None or self.threshold_current is None:
            return self.hyper_amp
        return self.threshold_current * (float(self.hyper_amp_rel) / 100.0)

    @property
    def depol_amplitude(self):
        if self.depol_amp_rel is None or self.threshold_current is None:
            return self.depol_amp
        return self.threshold_current * (float(self.depol_amp_rel) / 100.0)

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()

        self.time_vec.append(0.0)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(self.delay)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(self.delay)
        self.current_vec.append(self.holding_current + self.depol_amplitude)

        self.time_vec.append(self.tmid)
        self.current_vec.append(self.holding_current + self.depol_amplitude)

        self.time_vec.append(self.tmid)
        self.current_vec.append(self.holding_current + self.amplitude)

        self.time_vec.append(self.toff)
        self.current_vec.append(self.holding_current + self.amplitude)

        self.time_vec.append(self.toff)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(self.total_duration)
        self.current_vec.append(self.holding_current)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )

    def generate(self, dt=0.1):
        """Return current time series

        WARNING: do not offset ! This is on-top of a holding stimulus."""
        t = numpy.arange(0.0, self.total_duration, dt)
        current = numpy.full(t.shape, self.holding_current, dtype="float64")

        ton = int(self.delay / dt)
        tmid = int(self.tmid / dt)
        toff = int(self.toff / dt)

        current[ton:tmid] += self.depol_amplitude
        current[tmid:toff] += self.amplitude

        return t, current
