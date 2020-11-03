"""Validation functions."""

import logging

logger = logging.getLogger(__name__)


def validate(model):
    """ Decides if a models passes validation or not."""

    if max(model["scores"].values()) < 10.0:
        return True

    return False
