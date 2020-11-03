import logging
from bluepyopt.ephys.stimuli import Stimulus

logger = logging.getLogger(__name__)


class NrnHDPulse(Stimulus):
    def __init__(
        self,
        amp=None,
        amp2=None,
        ton=None,
        tmid=None,
        tmid2=None,
        toff=None,
        total_duration=None,
        location=None,
    ):
        """Constructor

        Args:
            location (Location): stimulus Location
        """

        super(NrnHDPulse, self).__init__()

        self.ton = ton
        self.step_amplitude = amp
        self.amp2 = amp2
        self.tmid = tmid
        self.step_delay = tmid
        self.tmid2 = tmid2
        self.toff = toff
        self.total_duration = total_duration
        self.location = location
        self.step_duration = self.toff - self.step_delay

        self.iclamp = None
        self.current_vec = None
        self.time_vec = None

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)
        logger.debug(
            "Adding HyperDepol stimulus to %s with delay %f, "
            "duration %f, and amplitude %f",
            str(self.location),
            self.step_delay,
            self.step_duration,
            self.step_amplitude,
        )

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()

        self.time_vec.append(0.0)
        self.current_vec.append(0.0)

        self.time_vec.append(self.ton)
        self.current_vec.append(0.0)

        self.time_vec.append(self.ton)
        self.current_vec.append(self.amp2)

        self.time_vec.append(self.tmid)
        self.current_vec.append(self.amp2)

        self.time_vec.append(self.tmid)
        self.current_vec.append(self.step_amplitude)

        self.time_vec.append(self.tmid2)
        self.current_vec.append(self.step_amplitude)

        self.time_vec.append(self.tmid2)
        self.current_vec.append(self.amp2)

        self.time_vec.append(self.toff)
        self.current_vec.append(self.amp2)

        self.time_vec.append(self.total_duration)
        self.current_vec.append(0.0)

        self.time_vec.append(self.total_duration)
        self.current_vec.append(0.0)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )

    def destroy(self, sim=None):
        """Destroy stimulus"""
        self.iclamp = None
        self.time_vec = None
        self.current_vec = None

    def __str__(self):
        """String representation"""

        return "Current play at %s" % (self.location)
