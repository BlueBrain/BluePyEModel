"""NegCheops stimulus class"""

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


class NegCheops(BPEM_stimulus):
    # pylint: disable=line-too-long,anomalous-backslash-in-string

    """NegCheops current stimulus

    .. code-block:: none

           holdi            holdi+amp              holdi               holdi+amp              holdi               holdi+amp              holdi
             :                  :                    :                     :                    :                     :                    :
             :                  :                    :                     :                    :                     :                    :
        |__________             :             ________________             :             ________________             :             ________________
        :          :\           :           /:                :\           :           /:                :\           :           /:                ^
        :          : \          :          / :                : \          :          / :                : \          :          / :                :
        :          :  \         :         /  :                :  \         :         /  :                :  \         :         /  :                :
        :          :   \        :        /   :                :   \        :        /   :                :   \        :        /   :                :
        :          :    \       :       /    :                :    \       :       /    :                :    \       :       /    :                :
        :          :     \      :      /     :                :     \      :      /     :                :     \      :      /     :                :
        :          :      \     :     /      :                :      \     :     /      :                :      \     :     /      :                :
        :          :       \    :    /       :                :       \    :    /       :                :       \    :    /       :                :
        :          :        \   :   /        :                :        \   :   /        :                :        \   :   /        :                :
        :          :         \  :  /         :                :         \  :  /         :                :         \  :  /         :                :
        :          :          \ : /          :                :          \ : /          :                :          \ : /          :                :
        :          :           \ /           :                :           \ /           :                :           \ /           :                :
        :          :            '            :                :            '            :                :            '            :                :
        :          :                         :                :                         :                :                         :                :
        t=0        delay                     t1               t2                        t3               t4                        toff   totduration
    """

    name = "NegCheops"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        self.amp = kwargs.get("amp", -0.250)
        self.amp_rel = kwargs.get("thresh_perc", None)

        self.holding_current = kwargs.get("holding_current", 0.0)
        self.threshold_current = None

        if self.amp is None and self.amp_rel is None:
            raise ValueError(f"In stimulus {self.name}, amp and thresh_perc cannot be both None.")

        if self.amplitude > self.holding_current:
            raise ValueError(
                f"Amplitude {self.amplitude} is supposed to be smaller than "
                + f"holding current {self.holding_current} in {self.name} stimulus."
            )

        self.delay = kwargs.get("delay", 1750.0)
        self.total_duration = kwargs.get("totduration", 18220.0)

        ramp1_duration = kwargs.get("ramp1_duration", 3333.0)
        ramp2_duration = kwargs.get("ramp2_duration", 1666.0)
        ramp3_duration = kwargs.get("ramp3_duration", 1111.0)
        inter_delay = kwargs.get("inter_delay", 2000.0)

        self.t1 = kwargs.get("t1", None)
        self.t2 = kwargs.get("t2", None)
        self.t3 = kwargs.get("t3", None)
        self.t4 = kwargs.get("t4", None)
        self.toff = kwargs.get("toff", None)

        if self.t1 is None and ramp1_duration is not None and self.delay is not None:
            self.t1 = self.delay + 2 * ramp1_duration
        if self.t2 is None and inter_delay is not None and self.t1 is not None:
            self.t2 = self.t1 + inter_delay
        if self.t3 is None and ramp2_duration is not None and self.t2 is not None:
            self.t3 = self.t2 + 2 * ramp2_duration
        if self.t4 is None and inter_delay is not None and self.t3 is not None:
            self.t4 = self.t3 + inter_delay
        if self.toff is None and ramp3_duration is not None and self.t4 is not None:
            self.toff = self.t4 + 2 * ramp3_duration

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
        self.time_vec.append((self.t1 + self.delay) / 2.0)
        self.current_vec.append(holding_current + self.amplitude)
        self.time_vec.append(self.t1)
        self.current_vec.append(holding_current)

        self.time_vec.append(self.t2)
        self.current_vec.append(holding_current)
        self.time_vec.append((self.t2 + self.t3) / 2.0)
        self.current_vec.append(holding_current + self.amplitude)
        self.time_vec.append(self.t3)
        self.current_vec.append(holding_current)

        self.time_vec.append(self.t4)
        self.current_vec.append(holding_current)
        self.time_vec.append((self.t4 + self.toff) / 2.0)
        self.current_vec.append(holding_current + self.amplitude)
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
        t1 = int(self.t1 / dt)
        t2 = int(self.t2 / dt)
        t3 = int(self.t3 / dt)
        t4 = int(self.t4 / dt)
        toff = int(self.toff / dt)

        mid = int(0.5 * (ton + t1))
        current[ton:mid] += numpy.linspace(0.0, self.amp, mid - ton + 1)[:-1]
        current[mid:t1] += numpy.linspace(self.amp, 0.0, t1 - mid + 1)[:-1]

        # Second peak
        mid = int(0.5 * (t2 + t3))
        current[t2:mid] += numpy.linspace(0.0, self.amp, mid - t2 + 1)[:-1]
        current[mid:t3] += numpy.linspace(self.amp, 0.0, t3 - mid + 1)[:-1]

        # Third peak
        mid = int(0.5 * (t4 + toff))
        current[t4:mid] += numpy.linspace(0.0, self.amp, mid - t4 + 1)[:-1]
        current[mid:toff] += numpy.linspace(self.amp, 0.0, toff - mid + 1)[:-1]

        return t, current
