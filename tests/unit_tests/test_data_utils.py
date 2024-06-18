"""Tests for EModelMetadata methods."""

"""
Copyright 2023-2024 Blue Brain Project / EPFL

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
import pytest

from bluepyemodel.data.utils import get_dendritic_data_filepath
from bluepyemodel.data.utils import read_dendritic_data


def test_get_dendritic_data_filepath():
    """Test get_dendritic_data_filepath function."""
    path1 = Path(get_dendritic_data_filepath("ISI_CV"))
    assert path1.is_file()
    assert path1.suffix == ".csv"

    path2 = Path(get_dendritic_data_filepath("rheobase"))
    assert path2.is_file()
    assert path2.suffix == ".csv"

    with pytest.raises(
        ValueError, match="read_data expects 'ISI_CV' or 'rheobase' but got bad_data_type"
    ):
        get_dendritic_data_filepath("bad_data_type")


def test_read_dendritic_data():
    """Test read_dendritic_data function."""
    dist, values = read_dendritic_data("ISI_CV")
    assert len(dist) == len(values) == 13
    assert dist[-1] == 496.43366619115545
    assert values[-1] == 0.8972286374133949

    dist, values = read_dendritic_data("rheobase")
    assert len(dist) == len(values) == 39
    assert dist[-1] == 394.66666666666674
    assert values[-1] == 0.30498175965665246
