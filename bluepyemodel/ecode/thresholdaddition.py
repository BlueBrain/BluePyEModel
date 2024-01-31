"""IDrest stimulus class"""

"""
Copyright 2023-2024, EPFL/Blue Brain Project

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

from .idrest import IDrest

logger = logging.getLogger(__name__)


class ThresholdAddition(IDrest):
    """IDrest current stimulus

    .. code-block:: none

              holdi               holdi+amp                holdi
                :                     :                      :
                :                     :                      :
                :           ______________________           :
                :          |                      |          :
                :          |                      |          :
                :          |                      |          :
                :          |                      |          :
        |__________________|                      |______________________
        ^                  ^                      ^                      ^
        :                  :                      :                      :
        :                  :                      :                      :
        t=0                delay                  delay+duration         totduration
    """

    name = "ThresholdAddition"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """
        super().__init__(location=location, **kwargs)
        if self.amp is None:
            raise TypeError(f"In stimulus {self.name}, amp cannot be None.")

    @property
    def amplitude(self):
        """Special amplitude: rheobase + self.amp"""
        if self.threshold_current is None:
            raise ValueError("threshold_current should not be None")
        return self.threshold_current + self.amp
