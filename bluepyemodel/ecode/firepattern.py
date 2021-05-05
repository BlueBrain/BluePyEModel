"""FirePattern stimulus class"""
import logging

from .idrest import IDrest

logger = logging.getLogger(__name__)


class FirePattern(IDrest):

    """FirePattern current stimulus"""

    name = "FirePattern"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        kwargs["thresh_perc"] = kwargs.get("thresh_perc", 200.0)

        kwargs["delay"] = kwargs.get("delay", 250.0)
        kwargs["duration"] = kwargs.get("duration", 3600.0)
        kwargs["totduration"] = kwargs.get("totduration", 4100.0)

        super().__init__(location=location, **kwargs)
