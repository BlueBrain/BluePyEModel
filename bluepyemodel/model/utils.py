"""Utils"""

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
