"""Utils"""

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

logger = logging.getLogger("__main__")


def temp_ljp_check(temperature, ljp_corrected, mech):
    """Returns True if mech has corresponding temperature and ljp correction

    Args:
        temperature (int): temperature associated with the mechanism if any
        ljp_corrected (bool): whether the mechanism is ljp corrected
        mech (MechanismConfiguration): mechanism
    """
    if temperature is None or temperature == mech.temperature:
        if ljp_corrected is None or ljp_corrected is mech.ljp_corrected:
            return True
    return False
