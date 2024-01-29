"""ProbAMPANMDA_EMS class"""

"""
Copyright 2023, EPFL/Blue Brain Project

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

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class ProbAMPANMDA_EMS(BPEM_stimulus):
    """ProbAMPANMDA_EMS synapse current injection"""

    name = "ProbAMPANMDA_EMS"

    def __init__(self, location=None, **kwargs):
        """Constructor

        Args:
            location (Location): stimulus Location
            syn_weight (float): synaptic weight
            syn_delay (float) synaptic delay (ms)
            total_duration (float): total duration (ms)
            use (float): synapse Use
            stimfreq (float): stimulus frequency (Hz)
            number (int): number of synaptic spikes
            noise (float): range 0 to 1. Fractional randomness
            netcon_thres (float): NetCon threshold (mV)
            netcon_delay (float): NetCon delay (ms)
            netcon_weight (float): NetCon weight
        """
        self.syn_weight = kwargs.get("syn_weight", None)
        self.syn_delay = kwargs.get("syn_delay", None)
        self.total_duration = kwargs.get("totduration", 100.0)
        self.use = kwargs.get("use", 1.0)
        self.stimfreq = kwargs.get("stimfreq", 70.0)
        self.number = kwargs.get("number", 1)
        self.noise = kwargs.get("noise", 0)
        self.netcon_thres = kwargs.get("netcon_thres", 10)
        self.netcon_delay = kwargs.get("netcon_delay", 0)
        self.netcon_weight = kwargs.get("netcon_weight", 700)
        self.synapse = None
        self.netstim = None
        self.netcon = None

        if self.syn_weight is None or self.syn_delay is None:
            raise TypeError(
                f"syn_weight and syn_delay should be specified for stimulus {self.name}"
            )

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.syn_delay

    @property
    def stim_end(self):
        return self.total_duration

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)
        logger.debug(
            "Adding synaptic excitatory stimulus to %s with delay %f and weight %f",
            str(self.location),
            self.syn_delay,
            self.syn_weight,
        )

        self.synapse = sim.neuron.h.ProbAMPANMDA_EMS(icomp.x, sec=icomp.sec)
        self.synapse.Use = self.use

        self.netstim = sim.neuron.h.NetStim(sec=icomp.sec)
        self.netstim.interval = 1000 / self.stimfreq
        self.netstim.number = self.number
        self.netstim.start = self.syn_delay
        self.netstim.noise = self.noise
        self.netcon = sim.neuron.h.NetCon(
            self.netstim,
            self.synapse,
            self.netcon_thres,
            self.netcon_delay,
            self.netcon_weight,
            sec=icomp.sec,
        )
        self.netcon.weight[0] = self.syn_weight

    def destroy(self, sim=None):
        """Destroy stimulus"""
        super().destroy(sim=sim)

        self.synapse = None
        self.netstim = None
        self.netcon = None
