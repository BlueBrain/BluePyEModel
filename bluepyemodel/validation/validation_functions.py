"""Validation functions."""

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

import numpy

logger = logging.getLogger(__name__)


def validate_max_score(model, threshold=5.0, validation_protocols_only=False):
    """Decides if a model passes validation or not."""

    if validation_protocols_only:
        return numpy.max(list(model.scores_validation.values())) < threshold

    values = list(model.scores.values()) + list(model.scores_validation.values())
    return numpy.max(values) < threshold


def validate_mean_score(model, threshold=5.0, validation_protocols_only=False):
    """Decides if a model passes validation or not."""

    if validation_protocols_only:
        return numpy.mean(list(model.scores_validation.values())) < threshold

    values = list(model.scores.values()) + list(model.scores_validation.values())
    return numpy.mean(values) < threshold
