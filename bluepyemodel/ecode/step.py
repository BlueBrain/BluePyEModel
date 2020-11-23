"""Step stimulus class"""
import logging
import numpy

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class Step(BPEM_stimulus):

    """Step current stimulus"""

    name = "Step"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        for k in ["amp", "delay", "duration", "totduration", "holding_current"]:
            if k not in kwargs:
                raise Exception(
                    "Argument {} missing for initialisation of " "eCode {}".format(k, self.name)
                )

        super().__init__(
            step_amplitude=kwargs["amp"],
            step_delay=kwargs["delay"],
            total_duration=kwargs["totduration"],
            step_duration=kwargs["duration"],
            holding_current=kwargs["holding_current"],
            location=location,
        )

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

        self.time_vec.append(self.step_delay)
        self.current_vec.append(self.holding_current + self.step_amplitude)

        self.time_vec.append(self.step_delay + self.step_duration)
        self.current_vec.append(self.holding_current + self.step_amplitude)

        self.time_vec.append(self.step_delay + self.step_duration)
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

        ton_idx = int(self.step_delay / dt)
        toff_idx = int((self.step_delay + self.step_duration) / dt)

        current[ton_idx:toff_idx] += self.step_amplitude

        return t, current
