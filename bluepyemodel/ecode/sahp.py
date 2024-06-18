"""sAHP stimulus class"""

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


class sAHP(BPEM_stimulus):
    """sAHP current stimulus

    The long step (here amp) is usually fixed at 40% of rheobase, and the short step (here amp2)
    can usually vary from 150% to 300% of rheobase.

    .. code-block:: none

           holdi       holdi+long_amp      holdi+amp       holdi+long_amp          holdi
             :                :                :                 :                   :
             :                :          ______________          :                   :
             :                :         |              |         :                   :
             :      ____________________|              |____________________         :
             :     |                    ^              ^                    |        :
             :     |                    :              :                    |        :
        |__________|                    :              :                    |__________________
        ^          ^                    :              :                    ^                  ^
        :          :                    :              :                    :                  :
        :          :                    :              :                    :                  :
        t=0        delay                tmid           tmid2                toff     totduration

    """

    name = "sAHP"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        self.amp = kwargs.get("amp", None)
        self.amp_rel = kwargs.get("thresh_perc", None)

        self.long_amp = kwargs.get("long_amp", None)
        self.long_amp_rel = kwargs.get("long_amp_rel", None)

        if self.amp is None and self.amp_rel is None:
            raise TypeError(f"In stimulus {self.name}, amp and thresh_perc cannot be both None.")

        if self.long_amp is None and self.long_amp_rel is None:
            raise TypeError(
                f"In stimulus {self.name}, long_amp and long_amp_rel cannot be both None."
            )

        self.holding_current = kwargs.get("holding_current", None)
        self.threshold_current = None

        self.delay = kwargs.get("delay", 250.0)
        self.tmid = kwargs.get("tmid", 500.0)
        self.tmid2 = kwargs.get("tmid2", 725.0)
        self.toff = kwargs.get("toff", 1175.0)
        self.total_duration = kwargs.get("totduration", 1425.0)

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
        if self.amp_rel is None or self.threshold_current is None:
            return self.amp
        return self.threshold_current * (float(self.amp_rel) / 100.0)

    @property
    def long_amplitude(self):
        if self.long_amp_rel is None or self.threshold_current is None:
            return self.long_amp
        return self.threshold_current * (float(self.long_amp_rel) / 100.0)

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

        self.time_vec.append(self.delay)
        self.current_vec.append(holding_current + self.long_amplitude)

        self.time_vec.append(self.tmid)
        self.current_vec.append(holding_current + self.long_amplitude)

        self.time_vec.append(self.tmid)
        self.current_vec.append(holding_current + self.amplitude)

        self.time_vec.append(self.tmid2)
        self.current_vec.append(holding_current + self.amplitude)

        self.time_vec.append(self.tmid2)
        self.current_vec.append(holding_current + self.long_amplitude)

        self.time_vec.append(self.toff)
        self.current_vec.append(holding_current + self.long_amplitude)

        self.time_vec.append(self.toff)
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
        """Return current time series

        WARNING: do not offset ! This is on-top of a holding stimulus."""
        holding_current = self.holding_current if self.holding_current is not None else 0

        t = numpy.arange(0.0, self.total_duration, dt)
        current = numpy.full(t.shape, holding_current, dtype="float64")

        ton = int(self.delay / dt)
        tmid = int(self.tmid / dt)
        tmid2 = int(self.tmid2 / dt)
        toff = int(self.toff / dt)

        current[ton:tmid] += self.long_amplitude
        current[tmid2:toff] += self.long_amplitude
        current[tmid:tmid2] += self.amplitude

        return t, current
