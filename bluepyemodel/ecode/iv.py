"""IV stimulus class"""
import logging

from .idrest import IDrest

logger = logging.getLogger(__name__)


class IV(IDrest):

    """IV current stimulus"""

    name = "IV"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        kwargs["thresh_perc"] = kwargs.get("thresh_perc", -40.0)

        kwargs["delay"] = kwargs.get("delay", 250.0)
        kwargs["duration"] = kwargs.get("duration", 3000.0)
        kwargs["totduration"] = kwargs.get("totduration", 3500.0)

        super().__init__(location=location, **kwargs)
