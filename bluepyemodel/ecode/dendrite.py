import numpy as np
from bluepyopt.ephys.locations import NrnSeclistCompLocation
from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation
from bluepyopt.ephys.locations import NrnTrunkSomaDistanceCompLocation

from .idrest import IDrest


class DendriticStep(IDrest):
    """Step protocol on a dendrite."""

    name = "DendriticStep"

    def __init__(self, location, **kwargs):
        """ """
        direction = kwargs.get("direction", "apical_trunk")
        if direction == "apical_trunk":
            sec_name = kwargs.get("seclist_name", "apical")

            if sec_name != "apical":
                raise Exception("With direction 'apical_trunk', sec_name must be apical")

            location = NrnTrunkSomaDistanceCompLocation(
                name="dend",
                soma_distance=kwargs.get("somadistance", 10),
                sec_index=kwargs.get("sec_index", None),
                sec_name="apical",
            )
        elif direction == "random":
            location = NrnSomaDistanceCompLocation(
                name="dend",
                soma_distance=kwargs.get("somadistance", 10),
                sec_name=kwargs.get("seclist_name", "apical"),
            )
        else:
            raise Exception(f"direction keyword {direction} not understood")

        super().__init__(location=location, **kwargs)

    def instantiate(self, sim=None, icell=None):
        """Force to have holding current at 0."""
        self.holding_current = 0
        super().instantiate(sim=sim, icell=icell)


class EPSP(IDrest):
    def __init__(self, location, **kwargs):
        """Constructor

        Args:
            step_amplitude (float): amplitude (nA)
            step_delay (float): delay (ms)
            step_duration (float): duration (ms)
            location (Location): stimulus Location
        """
        syn_location = NrnSeclistCompLocation(
            name="syn", comp_x=0.5, sec_index=kwargs.get("sec_index", 0), seclist_name="apical"
        )
        self.syn_delay = kwargs.get("syn_delay", 0.0)
        self.syn_amp = kwargs.get("syn_amp", 0.0)
        self.syn_duration = kwargs.get("syn_duration", 0.0)

        super().__init__(location=syn_location, **kwargs)

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""
        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()
        self.time_vec.append(0.0)
        self.current_vec.append(0)

        self.time_vec.append(self.delay + self.syn_delay)
        self.current_vec.append(0)

        t = np.linspace(0, self.total_duration - self.delay - self.syn_delay, 2000)
        rise = 0.5
        decay = 5.0
        s = np.exp(-t / decay) - np.exp(-t / rise)
        s = self.syn_amp * s / max(s)

        for _t, _s in zip(t, s):
            self.time_vec.append(self.delay + self.syn_delay + _t)
            self.current_vec.append(_s)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )


class BAC(IDrest):
    def __init__(self, location, **kwargs):
        """Constructor

        Args:
            step_amplitude (float): amplitude (nA)
            step_delay (float): delay (ms)
            step_duration (float): duration (ms)
            location (Location): stimulus Location
        """
        self.bap = IDrest(location, **kwargs)
        self.epsp = EPSP(None, **kwargs)

        super().__init__(location=None, **kwargs)

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""
        self.bap.holding_current = self.holding_current
        self.bap.threshold_current = self.threshold_current
        self.bap.instantiate(sim=sim, icell=icell)
        self.epsp.instantiate(sim=sim, icell=icell)
