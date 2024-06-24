"""Noise stimulus class"""

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

from bluepyemodel.ecode.stimulus import BPEM_stimulus

logger = logging.getLogger(__name__)


class NoiseMixin(BPEM_stimulus):
    """Noise current stimulus"""

    name = "Noise"

    def __init__(self, location):
        """Constructor

        Args:
            location(Location): location of stimulus
        """
        super().__init__(
            location=location,
        )

    @property
    def total_duration(self):
        return self.time_series[-1]

    @property
    def stim_start(self):
        return 0.0

    @property
    def stim_end(self):
        return self.time_series[-1]

    def generate(self, dt=0.1):
        """Return current time series"""
        holding_current = self.holding_current if self.holding_current is not None else 0

        if dt != 0.1:
            raise ValueError(f"For eCode {self.name}, dt has to be 0.1ms.")

        current = holding_current + self.current_series * (self.mu / 2.0) + self.mu

        return self.time_series, current
