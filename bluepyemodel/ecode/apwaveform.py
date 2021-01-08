"""APWaveform stimulus class"""
import logging

from .idrest import IDrest

logger = logging.getLogger(__name__)


class APWaveform(IDrest):

    """APWaveform current stimulus"""

    name = "APWaveform"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        kwargs["thresh_perc"] = kwargs.get("thresh_perc", 220.0)

        kwargs["delay"] = kwargs.get("delay", 250.0)
        kwargs["duration"] = kwargs.get("duration", 50.0)
        kwargs["totduration"] = kwargs.get("totduration", 550.0)

        super().__init__(location=location, **kwargs)
