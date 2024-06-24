"""Ecode for dendrite specific protocols, such as synaptic input, dendritic steps, or BAC."""

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

import numpy as np
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
                raise ValueError("With direction 'apical_trunk', sec_name must be apical")

            location = NrnTrunkSomaDistanceCompLocation(
                name="dend",
                soma_distance=kwargs["somadistance"],
                sec_index=kwargs.get("sec_index", None),
                seclist_name="apical",
            )
        elif direction == "random":
            location = NrnSomaDistanceCompLocation(
                name="dend",
                soma_distance=kwargs["somadistance"],
                seclist_name=kwargs.get("seclist_name", "apical"),
            )
        else:
            raise ValueError(f"direction keyword {direction} not understood")
        super().__init__(location=location, **kwargs)

    def instantiate(self, sim=None, icell=None):
        """Force to have holding current at 0."""
        self.holding_current = 0
        super().instantiate(sim=sim, icell=icell)


class Synaptic(DendriticStep):
    """Ecode to model a synapse with EPSP-like shape.

    A synthetic EPSP shape is defined by the difference of two exponentials, one with a
    rise time (syn_rise) constant, the other with a decay (syn_decay) time constants.
    It is normalized such that the maximum value is parametrized by syn_amp.
    """

    name = "Synaptic"

    def __init__(self, location, **kwargs):
        """Constructor

        Args:
            step_amplitude (float): amplitude (nA)
            step_delay (float): delay (ms)
            step_duration (float): duration (ms)
            location (Location): stimulus Location
            syn_delay (float): start time of synaptic input
            syn_amp (flaot): maximal amplitude of the synaptic input
            syn_rise (float): rise time constant
            syn_decay (float): decay time constant
        """
        super().__init__(location=None, **kwargs)

        self.syn_delay = kwargs.get("syn_delay", 0.0)
        self.syn_amp = kwargs.get("syn_amp", 0.0)
        self.syn_rise = kwargs.get("syn_rise", 0.5)
        self.syn_decay = kwargs.get("syn_decay", 5.0)

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
        s = np.exp(-t / self.syn_decay) - np.exp(-t / self.syn_rise)
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
    """BAC ecode.

    BAC is a combination of a bAP and a synaptic input to generate Ca dendritic spikes.
    """

    def __init__(self, location, **kwargs):
        """Constructor, combination of IDrest and Synaptic ecodes."""
        self.bap = IDrest(location, **kwargs)
        self.epsp = Synaptic(None, **kwargs)

        super().__init__(location=None, **kwargs)

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""
        self.bap.holding_current = self.holding_current
        self.bap.threshold_current = self.threshold_current
        self.bap.instantiate(sim=sim, icell=icell)
        self.epsp.instantiate(sim=sim, icell=icell)
