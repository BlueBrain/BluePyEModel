"""SpikeRec stimulus class"""
import logging

import numpy

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class SpikeRec(BPEM_stimulus):

    """SpikeRec current stimulus"""

    name = "SpikeRec"

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
            raise Exception(
                "In stimulus %s, amp and thresh_perc cannot be both None." % self.name
            )

        self.tspike = kwargs.get("tspike")
        self.spike_duration = kwargs.get("spike_duration")
        self.delta = kwargs.get("delta")
        self.total_duration = kwargs.get("totduration", 1850.0)

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.tspike[0]

    @property
    def stim_end(self):
        return (
            self.tspike[0]
            + len(self.tspike) * self.spike_duration
            + (len(self.tspike) - 1) * self.delta
        )

    @property
    def amplitude(self):
        if self.amp_rel is None or self.threshold_current is None:
            return self.amp
        return self.threshold_current * (float(self.amp_rel) / 100.0)

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()

        self.time_vec.append(0.0)
        self.current_vec.append(self.holding_current)

        spike_start = self.tspike[0]
        spike_end = spike_start + self.spike_duration

        self.time_vec.append(spike_start)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(spike_start)
        self.current_vec.append(self.holding_current + self.amplitude)

        self.time_vec.append(spike_end)
        self.current_vec.append(self.holding_current + self.amplitude)

        self.time_vec.append(spike_end)
        self.current_vec.append(self.holding_current)

        for _ in range(1, len(self.tspike)):
            spike_start = spike_end + self.delta
            spike_end = spike_start + self.spike_duration

            self.time_vec.append(spike_start)
            self.current_vec.append(self.holding_current)

            self.time_vec.append(spike_start)
            self.current_vec.append(self.holding_current + self.amplitude)

            self.time_vec.append(spike_end)
            self.current_vec.append(self.holding_current + self.amplitude)

            self.time_vec.append(spike_end)
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
        """Return current time series"""

        t = numpy.arange(0.0, self.total_duration, dt)
        current = numpy.full(t.shape, self.holding_current, dtype="float64")

        spike_start_idx = int(self.tspike[0] / dt)
        spike_end_idx = int((self.tspike[0] + self.spike_duration) / dt)
        current[spike_start_idx:spike_end_idx] += self.amplitude

        for _ in range(1, len(self.tspike)):
            spike_start_idx = int(spike_end_idx + (self.delta / dt))
            spike_end_idx = spike_start_idx + int(self.spike_duration / dt)
            current[spike_start_idx:spike_end_idx] += self.amplitude

        return t, current
