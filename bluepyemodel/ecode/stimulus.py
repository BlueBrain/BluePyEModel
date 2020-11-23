"""BPEM_stimulus class"""
import logging

from bluepyopt.ephys.stimuli import Stimulus

logger = logging.getLogger(__name__)


class BPEM_stimulus(Stimulus):

    """BPEM current stimulus"""

    name = ""

    def __init__(
        self, step_amplitude, step_delay, total_duration, step_duration, holding_current, location
    ):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        super().__init__()

        self.step_amplitude = step_amplitude
        self.step_delay = step_delay
        self.total_duration = total_duration
        self.step_duration = step_duration
        self.holding_current = holding_current
        self.location = location

        self.iclamp = None
        self.current_vec = None
        self.time_vec = None

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        time_series, current_series = self.generate(dt=0.1)

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = time_series[-1]

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()

        for t, i in zip(time_series, current_series):
            self.time_vec.append(t)
            self.current_vec.append(i)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )

    def destroy(self, sim=None):  # pylint:disable=W0613
        """Destroy stimulus"""

        self.iclamp = None
        self.time_vec = None
        self.current_vec = None

    def generate(self, dt=0.1):  # pylint:disable=W0613
        """Return current time series

        WARNING: do not offset ! This is on-top of a holding stimulus."""
        return [], []

    def __str__(self):
        """String representation"""

        return "%s current played at %s" % (self.name, self.location)
