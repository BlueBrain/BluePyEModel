"""BPEM_stimulus class"""

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

import logging

from bluepyopt.ephys.stimuli import Stimulus

logger = logging.getLogger(__name__)


class BPEM_stimulus(Stimulus):
    """Abstract current stimulus"""

    name = ""

    def __init__(self, location):
        """Constructor
        Args:
            total_duration (float): total duration of the stimulus in ms
            location(Location): location of stimulus
        """

        super().__init__()

        self.location = location

        self.iclamp = None
        self.current_vec = None
        self.time_vec = None

    @property
    def stim_start(self):
        return 0.0

    @property
    def stim_end(self):
        return 0.0

    @property
    def amplitude(self):
        return 0.0

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
        """Return current time series"""
        return [], []

    def __str__(self):
        """String representation"""
        return f"{self.name} current played at {self.location}"
