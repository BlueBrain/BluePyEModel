"""NegCheops stimulus class"""
import logging

import numpy

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class NegCheops(BPEM_stimulus):

    """NegCheops current stimulus"""

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
            raise Exception("In stimulus %s, amp and thresh_perc cannot be both None." % self.name)

        if self.amplitude > self.holding_current:
            raise Exception(
                f"Amplitude {self.amplitude} is supposed to be smaller than "
                + f"holding current {self.holding_current} in {self.name} stimulus."
            )

        self.delay = kwargs.get("delay", 1750.0)
        self.total_duration = kwargs.get("totduration", 16222.0)

        self.ramp1_duration = kwargs.get("ramp1_duration", 3333.0)
        self.ramp2_duration = kwargs.get("ramp2_duration", 1666.0)
        self.ramp3_duration = kwargs.get("ramp3_duration", 1111.0)
        self.inter_delay = kwargs.get("inter_delay", 2000.0)

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.delay

    @property
    def stim_end(self):
        return (
            self.delay
            + 2.0 * self.inter_delay
            + 2.0 * self.ramp1_duration
            + 2.0 * self.ramp2_duration
            + 2.0 * self.ramp3_duration
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

        self.time_vec.append(self.delay)
        self.current_vec.append(self.holding_current)
        self.time_vec.append(self.delay + self.ramp1_duration)
        self.current_vec.append(self.holding_current + self.amplitude)
        self.time_vec.append(self.delay + 2.0 * self.ramp1_duration)
        self.current_vec.append(self.holding_current)

        start_cheops2 = self.delay + 2.0 * self.ramp1_duration + self.inter_delay
        self.time_vec.append(start_cheops2)
        self.current_vec.append(self.holding_current)
        self.time_vec.append(start_cheops2 + self.ramp2_duration)
        self.current_vec.append(self.holding_current + self.amplitude)
        self.time_vec.append(start_cheops2 + 2.0 * self.ramp2_duration)
        self.current_vec.append(self.holding_current)

        start_cheops3 = start_cheops2 + 2.0 * self.ramp2_duration + self.inter_delay
        self.time_vec.append(start_cheops3)
        self.current_vec.append(self.holding_current)
        self.time_vec.append(start_cheops3 + self.ramp3_duration)
        self.current_vec.append(self.holding_current + self.amplitude)
        self.time_vec.append(start_cheops3 + 2.0 * self.ramp3_duration)
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

        idx_ton = int(self.delay / dt)
        idx_inter_delay = int(self.inter_delay / dt)
        idx_ramp1_duration = int(self.ramp1_duration / dt)
        idx_ramp2_duration = int(self.ramp2_duration / dt)
        idx_ramp3_duration = int(self.ramp3_duration / dt)

        current[idx_ton : idx_ton + idx_ramp1_duration] += numpy.linspace(
            0.0, self.amplitude, idx_ramp1_duration + 1
        )[:-1]
        current[
            idx_ton + idx_ramp1_duration : idx_ton + (2 * idx_ramp1_duration)
        ] += numpy.linspace(self.amplitude, 0.0, idx_ramp1_duration + 1)[:-1]

        idx_ton2 = idx_ton + (2 * idx_ramp1_duration) + idx_inter_delay
        current[idx_ton2 : idx_ton2 + idx_ramp2_duration] += numpy.linspace(
            0.0, self.amplitude, idx_ramp2_duration + 1
        )[:-1]
        current[
            idx_ton2 + idx_ramp2_duration : idx_ton2 + (2 * idx_ramp2_duration)
        ] += numpy.linspace(self.amplitude, 0.0, idx_ramp2_duration + 1)[:-1]

        idx_ton3 = idx_ton2 + (2 * idx_ramp2_duration) + idx_inter_delay
        current[idx_ton3 : idx_ton3 + idx_ramp3_duration] += numpy.linspace(
            0.0, self.amplitude, idx_ramp3_duration + 1
        )[:-1]
        current[
            idx_ton3 + idx_ramp3_duration : idx_ton3 + (2 * idx_ramp3_duration)
        ] += numpy.linspace(self.amplitude, 0.0, idx_ramp3_duration + 1)[:-1]

        return t, current
