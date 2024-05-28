"""Tests for EModelMetadata methods."""

"""
Copyright 2024, EPFL/Blue Brain Project

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

import numpy
import pytest

from bluepyemodel.model.morphology_utils import get_apical_length
from bluepyemodel.model.morphology_utils import get_apical_point_soma_distance
from bluepyemodel.model.morphology_utils import get_basal_and_apical_lengths
from bluepyemodel.model.morphology_utils import get_hotspot_location

from tests.utils import DATA


@pytest.fixture
def morph_path():
    return DATA / "morphology" / "C060114A5.asc"


def test_get_apical_point_soma_distance(morph_path):
    """Test get_apical_point_soma_distance function."""
    soma_dist = get_apical_point_soma_distance(morph_path)
    numpy.testing.assert_allclose(soma_dist, 624.8454)


def test_get_apical_length(morph_path):
    """Test get_apical_length function."""
    apical_length =  get_apical_length(morph_path)
    numpy.testing.assert_allclose(apical_length, 1044.1445)


def test_get_basal_and_apical_lengths(morph_path):
    """Test get_basal_and_apical_lengths function."""
    basal_length, apical_length = get_basal_and_apical_lengths(morph_path)
    numpy.testing.assert_allclose(apical_length, 1044.1445)
    numpy.testing.assert_allclose(basal_length, 232.56221)


def test_get_hotspot_location(morph_path):
    """Test get_hotspot_location function."""
    hotspot_start, hotspot_end = get_hotspot_location(morph_path)
    numpy.testing.assert_allclose(hotspot_start, 520.430945)
    numpy.testing.assert_allclose(hotspot_end, 729.259851)


def test_morphology_utils(morph_path):
    """Test that output of morphology_utils functions are consistent with one another."""
    apical_length_1 =  get_apical_length(morph_path)
    basal_length, apical_length_2 = get_basal_and_apical_lengths(morph_path)
    soma_dist = get_apical_point_soma_distance(morph_path)
    hotspot_start, hotspot_end = get_hotspot_location(morph_path)

    assert apical_length_1 == apical_length_2
    assert basal_length < apical_length_2
    assert soma_dist < apical_length_1
    assert hotspot_start >= 0
    assert hotspot_start < soma_dist < hotspot_end
