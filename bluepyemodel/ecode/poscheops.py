"""PosCheops stimulus class"""
import logging
import numpy

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class PosCheops(BPEM_stimulus):

    """PosCheops current stimulus"""

    name = "PosCheops"

    ramp1_duration = 4000.0
    ramp2_duration = 2000.0
    ramp3_duration = 1333.0
    inter_delay = 500.0

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        for k in ["delay", "amp", "holding_current"]:
            if k not in kwargs:
                raise Exception(
                    "Argument {} missing for initialisation of " "eCode {}".format(k, self.name)
                )

        total_duration = (
            kwargs["delay"]
            + 2.0 * self.ramp1_duration
            + 2.0 * self.ramp2_duration
            + 2.0 * self.ramp3_duration
            + 3.0 * self.inter_delay
        )
        step_duration = total_duration - self.inter_delay - kwargs["delay"]

        super().__init__(
            step_amplitude=kwargs["amp"],
            step_delay=kwargs["delay"],
            total_duration=total_duration,
            step_duration=step_duration,
            holding_current=kwargs["holding_current"],
            location=location,
        )

        self.toff = self.step_duration + self.step_delay

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()

        self.time_vec.append(0.0)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(self.step_delay)
        self.current_vec.append(self.holding_current)
        self.time_vec.append(self.step_delay + self.ramp1_duration)
        self.current_vec.append(self.holding_current + self.step_amplitude)
        self.time_vec.append(self.step_delay + 2.0 * self.ramp1_duration)
        self.current_vec.append(self.holding_current)

        start_cheops2 = self.step_delay + 2.0 * self.ramp1_duration + self.inter_delay
        self.time_vec.append(start_cheops2)
        self.current_vec.append(self.holding_current)
        self.time_vec.append(start_cheops2 + self.ramp2_duration)
        self.current_vec.append(self.holding_current + self.step_amplitude)
        self.time_vec.append(start_cheops2 + 2.0 * self.ramp2_duration)
        self.current_vec.append(self.holding_current)

        start_cheops3 = start_cheops2 + 2.0 * self.ramp2_duration + self.inter_delay
        self.time_vec.append(start_cheops3)
        self.current_vec.append(self.holding_current)
        self.time_vec.append(start_cheops3 + self.ramp3_duration)
        self.current_vec.append(self.holding_current + self.step_amplitude)
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

        idx_ton = int(self.step_delay / dt)
        idx_inter_delay = int(self.inter_delay / dt)
        idx_ramp1_duration = int(self.ramp1_duration / dt)
        idx_ramp2_duration = int(self.ramp2_duration / dt)
        idx_ramp3_duration = int(self.ramp3_duration / dt)

        current[idx_ton:idx_ton + idx_ramp1_duration] += numpy.linspace(
            0.0, self.step_amplitude, idx_ramp1_duration
        )
        current[
            idx_ton + idx_ramp1_duration:idx_ton + (2 * idx_ramp1_duration)
        ] += numpy.linspace(self.step_amplitude, 0.0, idx_ramp1_duration)

        idx_ton2 = idx_ton + (2 * idx_ramp1_duration) + idx_inter_delay
        current[idx_ton2:idx_ton2 + idx_ramp2_duration] += numpy.linspace(
            0.0, self.step_amplitude, idx_ramp2_duration
        )
        current[
            idx_ton2 + idx_ramp2_duration:idx_ton2 + (2 * idx_ramp2_duration)
        ] += numpy.linspace(self.step_amplitude, 0.0, idx_ramp2_duration)

        idx_ton3 = idx_ton2 + (2 * idx_ramp2_duration) + idx_inter_delay
        current[idx_ton3:idx_ton3 + idx_ramp3_duration] += numpy.linspace(
            0.0, self.step_amplitude, idx_ramp3_duration
        )
        current[
            idx_ton3 + idx_ramp3_duration:idx_ton3 + (2 * idx_ramp3_duration)
        ] += numpy.linspace(self.step_amplitude, 0.0, idx_ramp3_duration)

        return t, current
