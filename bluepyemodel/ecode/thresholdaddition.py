"""IDrest stimulus class"""
import logging

import numpy

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
        super().__init__(
            location=location, **kwargs
        )
        if self.amp is None:
            raise TypeError(f"In stimulus {self.name}, amp cannot be None.")

    @property
    def amplitude(self):
        """Special amplitude: rheobase + self.amp"""
        if self.threshold_current is None:
            raise ValueError("threshold_current should not be None")
        return self.threshold_current + self.amp
