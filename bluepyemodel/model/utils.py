"""Utils"""

import logging

logger = logging.getLogger("__main__")


def temperature_check(temperature, mech):
    """Returns True if temperature is None or mech has corresponding temperature

    Args:
        temperature (int): temperature associated with the mechanism if any
        mech (MechanismConfiguration): mechanism
    """
    if temperature is None:
        return True
    if temperature == mech.temperature:
        return True
    return False
