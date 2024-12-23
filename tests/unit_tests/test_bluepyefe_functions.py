"""Tests for BluePyEfe functions. So that a change in BPE does not go unnoticed."""

"""
Copyright 2024 Blue Brain Project / EPFL

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

from pathlib import Path
import bluepyefe.extract


def test_cells_pickle_output_path():
    """Test cells_pickle_output_path function."""
    assert bluepyefe.extract.cells_pickle_output_path(Path("output")) == Path("output/cells.pkl")


def test_protocols_pickle_output_path():
    """Test protocols_pickle_output_path function."""
    assert bluepyefe.extract.protocols_pickle_output_path(Path("output")) == Path("output/protocols.pkl")
