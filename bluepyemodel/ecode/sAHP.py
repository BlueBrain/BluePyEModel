"""sAHP stimulus class"""
import logging
import numpy

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class sAHP(BPEM_stimulus):

    """sAHP current stimulus"""

    name = "sAHP"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        for k in [
            "delay",
            "totduration",
            "amp",
            "tmid",
            "tmid2",
            "toff",
            "long_amp",
            "holding_current",
        ]:
            if k not in kwargs:
                raise Exception(
                    "Argument {} missing for initialisation of " "eCode {}".format(k, self.name)
                )

        self.tmid = kwargs["tmid"]
        self.tmid2 = kwargs["tmid2"]
        self.toff = kwargs["toff"]
        self.long_amp = kwargs["long_amp"]

        super().__init__(
            step_amplitude=kwargs["amp"],
            step_delay=kwargs["delay"],
            total_duration=kwargs["totduration"],
            step_duration=self.tmid2 - self.tmid,
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
        self.current_vec.append(self.holding_current + self.long_amp)

        self.time_vec.append(self.tmid)
        self.current_vec.append(self.holding_current + self.long_amp)

        self.time_vec.append(self.tmid)
        self.current_vec.append(self.holding_current + self.step_amplitude)

        self.time_vec.append(self.tmid2)
        self.current_vec.append(self.holding_current + self.step_amplitude)

        self.time_vec.append(self.tmid2)
        self.current_vec.append(self.holding_current + self.long_amp)

        self.time_vec.append(self.toff)
        self.current_vec.append(self.holding_current + self.long_amp)

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

        ton = int(self.step_delay / dt)
        tmid = int(self.tmid / dt)
        tmid2 = int(self.tmid2 / dt)
        toff = int(self.toff / dt)

        current[ton:tmid] += self.long_amp
        current[tmid2:toff] += self.long_amp
        current[tmid:tmid2] += self.step_amplitude

        return t, current
