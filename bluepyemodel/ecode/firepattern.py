"""FirePattern stimulus class"""

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
