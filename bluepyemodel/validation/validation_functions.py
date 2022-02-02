"""Validation functions."""

import logging

import numpy

logger = logging.getLogger(__name__)


def validate_max_score(model, threshold=5.0, validation_protocols_only=False):
    """Decides if a model passes validation or not."""

    if validation_protocols_only:
        return max(model.scores_validation.values()) < threshold

    return max(model.scores.values()) < threshold


def validate_mean_score(model, threshold=5.0, validation_protocols_only=False):
    """Decides if a model passes validation or not."""

    if validation_protocols_only:
        return numpy.mean(list(model.scores_validation.values())) < threshold

    return numpy.mean(list(model.scores.values())) < threshold
