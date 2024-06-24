"""SpikeRec stimulus class"""

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


class SpikeRecMultiSpikes(BPEM_stimulus):
    # pylint: disable=line-too-long

    """SpikeRecMultiSpikes current stimulus

    .. code-block:: none

              holdi        holdi+amp        holdi       holdi+amp            .   .   .
                :               :             :             :
                :       _________________     :      _________________                      _________________
                :      |                 |    :     |                 |                    |                 |
                :      |                 |    :     |                 |     * n_spikes     |                 |
                :      |                 |    :     |                 |     .   .   .      |                 |
                :      |                 |    :     |                 |                    |                 |
        |______________|                 |__________|                 |__                __|                 |___
        :              :                 :          :                 :                                          ^
        :              :                 :          :                 :                                          :
        :              :                 :          :                 :                                          :
         <--  delay  --><-spike_duration-><- delta -><-spike_duration->     .   .   .                  totduration

    """

    name = "SpikeRecMultiSpikes"

    def __init__(self, location, **kwargs):
        """Constructor

        Attention! This is the class for the new SpikeRec containing multispikes.
        In order to use it, you should use ``SpikeRecMultiSpikes`` as key in your protocol file.
        If you use ``SpikeRec``, it will use the old SpikeRec containing one spike,
        which is using the IDRest class.
        Beware that the `**kwargs` for the two types (multispikes/1spike) of SpikeRec are different.

        Args:
            location(Location): location of stimulus
            **kwargs: See below

        Keyword Arguments:
            amp (float): amplitude of each spike(nA)
            thresh_perc (float): amplitude of each spike relative
                to the threshold current (%)
            holding_current (float): amplitude of the holding current (nA)
            delay (float): time at which the first current spike begins (ms)
            n_spikes (int): number of spikes for the stimulus
            spike_duration (float): duration of each spike (ms)
            delta (float): time without stimulus between each spike (ms)
            totduration (float): total duration of the whole stimulus (ms)
        """

        self.amp = kwargs.get("amp", None)
        self.amp_rel = kwargs.get("thresh_perc", None)

        self.holding_current = kwargs.get("holding_current", None)
        self.threshold_current = None

        if self.amp is None and self.amp_rel is None:
            raise TypeError(f"In stimulus {self.name}, amp and thresh_perc cannot be both None.")

        self.delay = kwargs.get("delay", 10.0)
        self.n_spikes = kwargs.get("n_spikes", 2)
        self.spike_duration = kwargs.get("spike_duration", 3.5)
        self.delta = kwargs.get("delta", 3.5)
        self.total_duration = kwargs.get("totduration", 1500.0)

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.delay

    @property
    def stim_end(self):
        return self.delay + self.n_spikes * self.spike_duration + (self.n_spikes - 1) * self.delta

    @property
    def amplitude(self):
        if self.amp_rel is None or self.threshold_current is None:
            return self.amp
        return self.threshold_current * (float(self.amp_rel) / 100.0)

    def multi_stim_start(self):
        return [self.delay + i * (self.spike_duration + self.delta) for i in range(self.n_spikes)]

    def multi_stim_end(self):
        return [ss + self.spike_duration for ss in self.multi_stim_start()]

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

        spike_start = self.delay
        spike_end = spike_start + self.spike_duration

        self.time_vec.append(spike_start)
        self.current_vec.append(holding_current)

        self.time_vec.append(spike_start)
        self.current_vec.append(holding_current + self.amplitude)

        self.time_vec.append(spike_end)
        self.current_vec.append(holding_current + self.amplitude)

        self.time_vec.append(spike_end)
        self.current_vec.append(holding_current)

        for _ in range(1, self.n_spikes):
            spike_start = spike_end + self.delta
            spike_end = spike_start + self.spike_duration

            self.time_vec.append(spike_start)
            self.current_vec.append(holding_current)

            self.time_vec.append(spike_start)
            self.current_vec.append(holding_current + self.amplitude)

            self.time_vec.append(spike_end)
            self.current_vec.append(holding_current + self.amplitude)

            self.time_vec.append(spike_end)
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

        spike_start_idx = int(self.delay / dt)
        spike_end_idx = int((self.delay + self.spike_duration) / dt)
        current[spike_start_idx:spike_end_idx] += self.amplitude

        for _ in range(1, self.n_spikes):
            spike_start_idx = int(spike_end_idx + (self.delta / dt))
            spike_end_idx = spike_start_idx + int(self.spike_duration / dt)
            current[spike_start_idx:spike_end_idx] += self.amplitude

        return t, current
