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

import pytest
from unittest.mock import PropertyMock, patch, Mock
from bluepyemodel.access_point.forge_access_point import NexusForgeAccessPoint, AccessPointException
from bluepyemodel.access_point.nexus import NexusAccessPoint
from datetime import datetime, timezone, timedelta
import logging


def jwt_payload():
    return {
        "preferred_username": "test_user",
        "name": "Test User",
        "email": "test_user@example.com",
        "sub": "test_sub",
        "exp": (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()
    }


@pytest.fixture(autouse=True)
def mock_jwt_decode():
    with patch("jwt.decode") as mock_jwt:
        mock_jwt.return_value = jwt_payload()
        yield mock_jwt


@pytest.fixture
def mock_nexus_access_point():
    with patch("bluepyemodel.access_point.forge_access_point.NexusForgeAccessPoint.refresh_token") as mock_refresh, \
         patch.object(NexusForgeAccessPoint, "forge", new_callable=PropertyMock) as mock_forge_prop, \
         patch("bluepyemodel.access_point.nexus.get_brain_region_notation", return_value="SS") as mock_brain_region, \
         patch("bluepyemodel.access_point.nexus.NexusAccessPoint.get_pipeline_settings", return_value=Mock()) as mock_pipeline_settings, \
         patch("bluepyemodel.access_point.nexus.NexusAccessPoint.build_ontology_based_metadata") as mock_build_metadata:

        mock_refresh.return_value = (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()

        mock_forge = Mock()
        mock_resource = Mock()
        mock_resource.id = '0'
        mock_forge.resolve.return_value = mock_resource
        mock_forge_prop.return_value = mock_forge

        mock_nexus_forge_access_point = NexusForgeAccessPoint(
            project="test",
            organisation="demo",
            endpoint="https://bbp.epfl.ch/nexus/v1",
            forge_path=None,
            access_token="test_token"
        )

        with patch("bluepyemodel.access_point.forge_access_point.NexusForgeAccessPoint", return_value=mock_nexus_forge_access_point):
            yield NexusAccessPoint(
                emodel="L5_TPC",
                etype="cAC",
                ttype="189_L4/5 IT CTX",
                mtype="L5_TPC:B",
                species="mouse",
                brain_region="SSCX",
                iteration_tag="v0",
                project="test",
                organisation="demo",
                endpoint="https://bbp.epfl.ch/nexus/v1",
                forge_path=None,
                forge_ontology_path=None,
                access_token="test_token",
                sleep_time=0
            )


@pytest.fixture
def nexus_patches():
    with patch("bluepyemodel.access_point.nexus.get_brain_region_notation", return_value="SS") as mock_brain_region, \
         patch("bluepyemodel.access_point.nexus.NexusAccessPoint.get_pipeline_settings", return_value=Mock()) as mock_pipeline_settings, \
         patch("bluepyemodel.access_point.nexus.NexusAccessPoint.build_ontology_based_metadata") as mock_build_metadata:
        yield mock_brain_region, mock_pipeline_settings, mock_build_metadata


def test_init(mock_nexus_access_point):
    """
    Test the initialization of the NexusAccessPoint.
    """
    emodel_metadata = mock_nexus_access_point.emodel_metadata
    assert emodel_metadata.emodel == "L5_TPC"
    assert emodel_metadata.etype == "cAC"
    assert emodel_metadata.ttype == "189_L4/5 IT CTX"
    assert emodel_metadata.mtype == "L5_TPC:B"
    assert emodel_metadata.species == "mouse"
    assert emodel_metadata.brain_region == "SSCX"
    assert emodel_metadata.allen_notation == "SS"
    assert emodel_metadata.iteration == "v0"
    assert mock_nexus_access_point.forge_ontology_path is None
    assert mock_nexus_access_point.sleep_time == 0

    resolved_resource = mock_nexus_access_point.access_point.forge.resolve()
    assert resolved_resource.id == '0'


@pytest.fixture
def mock_available_etypes():
    with patch.object(NexusForgeAccessPoint, 'available_etypes', new_callable=PropertyMock) as mock_etypes:
        mock_etypes.return_value = ["0", "1", "2"]
        yield mock_etypes


@pytest.fixture
def mock_available_mtypes():
    with patch.object(NexusForgeAccessPoint, 'available_mtypes', new_callable=PropertyMock) as mock_mtypes:
        mock_mtypes.return_value = ["0", "1", "2"]
        yield mock_mtypes


@pytest.fixture
def mock_available_ttypes():
    with patch.object(NexusForgeAccessPoint, 'available_ttypes', new_callable=PropertyMock) as mock_ttypes:
        mock_ttypes.return_value = ["0", "1", "2"]
        yield mock_ttypes


@pytest.fixture
def mock_check_mettypes_dependencies():
    with patch("bluepyemodel.access_point.nexus.ontology_forge_access_point") as mock_ontology_forge, \
         patch("bluepyemodel.access_point.nexus.check_resource") as mock_check_resource:
        mock_ontology_forge.return_value = Mock()
        yield mock_ontology_forge, mock_check_resource


def test_check_mettypes(mock_nexus_access_point, mock_available_etypes, mock_available_mtypes, mock_available_ttypes, mock_check_mettypes_dependencies, caplog):
    """
    Test the check_mettypes function of the NexusAccessPoint.
    """
    mock_ontology_forge, mock_check_resource = mock_check_mettypes_dependencies

    mock_nexus_access_point.emodel_metadata.etype = "cAC"
    mock_nexus_access_point.emodel_metadata.mtype = None
    mock_nexus_access_point.emodel_metadata.ttype = None

    with caplog.at_level(logging.INFO):
        mock_nexus_access_point.check_mettypes()

    assert "Checking if etype cAC is present on nexus..." in caplog.text
    assert "Etype checked" in caplog.text
    assert "Mtype is None, its presence on Nexus is not being checked." in caplog.text
    assert "Ttype is None, its presence on Nexus is not being checked." in caplog.text

    mock_ontology_forge.assert_called_once_with(
        mock_nexus_access_point.access_point.access_token,
        mock_nexus_access_point.forge_ontology_path,
        mock_nexus_access_point.access_point.endpoint,
    )

    mock_check_resource.assert_any_call(
        "cAC",
        "etype",
        access_point=mock_ontology_forge.return_value,
        access_token=mock_nexus_access_point.access_point.access_token,
        forge_path=mock_nexus_access_point.forge_ontology_path,
        endpoint=mock_nexus_access_point.access_point.endpoint,
    )


def test_get_nexus_subject_none(mock_nexus_access_point):
    """
    Test get_nexus_subject with None as input.
    """
    assert mock_nexus_access_point.get_nexus_subject(None) is None


def test_get_nexus_subject_human(mock_nexus_access_point):
    """
    Test get_nexus_subject with 'human' as input.
    """
    expected_subject = {
        "type": "Subject",
        "species": {
            "id": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
            "label": "Homo sapiens",
        },
    }
    assert mock_nexus_access_point.get_nexus_subject("human") == expected_subject


@pytest.mark.parametrize("species, expected_subject", [
    (None, None),
    ("human", {
        "type": "Subject",
        "species": {
            "id": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
            "label": "Homo sapiens",
        }
    }),
    ("mouse", {
        "type": "Subject",
        "species": {
            "id": "http://purl.obolibrary.org/obo/NCBITaxon_10090",
            "label": "Mus musculus",
        }
    }),
    ("rat", {
        "type": "Subject",
        "species": {
            "id": "http://purl.obolibrary.org/obo/NCBITaxon_10116",
            "label": "Rattus norvegicus",
        }
    }),
])


def test_get_nexus_subject_parametrized(mock_nexus_access_point, species, expected_subject):
    """
    Parametrized test for get_nexus_subject with different species inputs.
    """
    assert mock_nexus_access_point.get_nexus_subject(species) == expected_subject


def test_get_nexus_subject_unknown_species(mock_nexus_access_point):
    """
    Test get_nexus_subject with an unknown species input.
    """
    with pytest.raises(ValueError, match="Unknown species unknown_species."):
        mock_nexus_access_point.get_nexus_subject("unknown_species")
