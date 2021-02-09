"""Validation functions."""

import logging

import numpy

logger = logging.getLogger(__name__)


def max_score(model, threshold=5.0, validation_protocols_only=False):
    """ Decides if a models passes validation or not."""

    if validation_protocols_only:
        if max(model["scores_validation"].values()) < threshold:
            return True
    else:
        if max(model["scores"].values()) < threshold:
            return True

    return False


def mean_score(model, threshold=5.0, validation_protocols_only=False):
    """ Decides if a models passes validation or not."""

    if validation_protocols_only:
        if numpy.mean(model["scores_validation"].values()) < threshold:
            return True
    else:
        if numpy.mean(model["scores"].values()) < threshold:
            return True

    return False
