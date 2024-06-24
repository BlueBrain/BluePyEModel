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

import math
import pytest
from unittest.mock import Mock, patch
from bluepyemodel.access_point.forge_access_point import AccessPointException, NexusForgeAccessPoint, get_brain_region, get_brain_region_dict
from datetime import datetime, timezone, timedelta

@pytest.fixture
def mock_forge_access_point():
    with patch("jwt.decode") as mock_jwt_decode:
        mock_jwt_decode.return_value = {
            "preferred_username": "test_user",
            "name": "Test User",
            "email": "test_user@example.com",
            "sub": "test_sub",
            "exp": (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()
        }
        with patch("bluepyemodel.access_point.forge_access_point.NexusForgeAccessPoint.refresh_token") as mock_refresh_token:
            mock_refresh_token.return_value = (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()
            with patch("bluepyemodel.access_point.forge_access_point.KnowledgeGraphForge") as mock_kg_forge:
                return NexusForgeAccessPoint(
                    project="test",
                    organisation="demo",
                    endpoint="https://bbp.epfl.ch/nexus/v1",
                    forge_path=None,
                    access_token="test_token"
                )


def test_nexus_forge_access_point_init(mock_forge_access_point):
    """
    Test the initialization of NexusForgeAccessPoint.
    """
    assert mock_forge_access_point.bucket == "demo/test"
    assert mock_forge_access_point.endpoint == "https://bbp.epfl.ch/nexus/v1"
    assert mock_forge_access_point.access_token == "test_token"
    assert mock_forge_access_point.agent.id == "https://bbp.epfl.ch/nexus/v1/realms/bbp/users/test_user"


def test_refresh_token_not_expired(mock_forge_access_point):
    """
    Test refresh_token method when the token is not expired.
    """
    future_exp = (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()
    with patch("jwt.decode") as mock_jwt_decode:
        mock_jwt_decode.return_value = {
            "preferred_username": "test_user",
            "name": "Test User",
            "email": "test_user@example.com",
            "sub": "test_sub",
            "exp": future_exp
        }
        exp_timestamp = mock_forge_access_point.refresh_token()
        assert math.isclose(exp_timestamp, future_exp, abs_tol=0.1)


def test_refresh_token_expired_offset(mock_forge_access_point, caplog):
    """
    Test refresh_token method when the token is about to expire.
    """
    future_exp = (datetime.now(timezone.utc) + timedelta(seconds=299)).timestamp()
    new_future_exp = (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()
    with patch("jwt.decode") as mock_jwt_decode:
        mock_jwt_decode.side_effect = [
            {
                "preferred_username": "test_user",
                "name": "Test User",
                "email": "test_user@example.com",
                "sub": "test_sub",
                "exp": future_exp
            },
            {
                "preferred_username": "test_user",
                "name": "Test User",
                "email": "test_user@example.com",
                "sub": "test_sub",
                "exp": new_future_exp
            }
        ]
        with patch.object(mock_forge_access_point, "get_access_token", return_value="new_test_token"):
            with caplog.at_level("INFO"):
                exp_timestamp = mock_forge_access_point.refresh_token()
                assert math.isclose(exp_timestamp, new_future_exp, abs_tol=0.1)
                assert "Nexus access token has expired, refreshing token..." in caplog.text


def test_refresh_token_expired(mock_forge_access_point, caplog):
    """
    Test refresh_token method when the token has expired.
    """
    past_exp = (datetime.now(timezone.utc) - timedelta(seconds=1)).timestamp()
    new_future_exp = (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()
    with patch("jwt.decode") as mock_jwt_decode:
        mock_jwt_decode.side_effect = [
            {
                "preferred_username": "test_user",
                "name": "Test User",
                "email": "test_user@example.com",
                "sub": "test_sub",
                "exp": past_exp
            },
            {
                "preferred_username": "test_user",
                "name": "Test User",
                "email": "test_user@example.com",
                "sub": "test_sub",
                "exp": new_future_exp
            }
        ]
        with patch.object(mock_forge_access_point, "get_access_token", return_value="new_test_token"):
            with caplog.at_level("INFO"):
                exp_timestamp = mock_forge_access_point.refresh_token()
                assert math.isclose(exp_timestamp, new_future_exp, abs_tol=0.1)
                assert "Nexus access token has expired, refreshing token..." in caplog.text


@pytest.fixture
def mock_get_brain_region_resolve():
    with patch("bluepyemodel.access_point.forge_access_point.ontology_forge_access_point") as mock_ontology_access_point:
        mock_access_point = Mock()
        mock_ontology_access_point.return_value = mock_access_point

        def resolve(brain_region, strategy, **kwargs):
            if brain_region.lower() in ["somatosensory areas", "basic cell groups and regions", "mock_brain_region"]:
                mock_resource = Mock()
                mock_resource.id = "mock_id"
                mock_resource.label = "mock_label"
                return mock_resource
            return None

        mock_access_point.resolve = resolve
        yield mock_ontology_access_point, mock_access_point


def test_get_brain_region_found(mock_get_brain_region_resolve):
    """
    Test get_brain_region function when the brain region is found.
    """
    resource = get_brain_region("SSCX", access_token="test_token")
    assert resource.id == "mock_id"
    assert resource.label == "mock_label"


def test_get_brain_region_not_found(mock_get_brain_region_resolve):
    """
    Test get_brain_region function when the brain region is not found.
    """
    with pytest.raises(AccessPointException, match=r"Could not find any brain region with name UnknownRegion"):
        get_brain_region("UnknownRegion", access_token="test_token")


def test_get_brain_region_dict(mock_get_brain_region_resolve):
    """
    Test get_brain_region_dict function to ensure it returns the correct dictionary.
    """
    _, mock_access_point = mock_get_brain_region_resolve

    mock_access_point.forge.as_json.return_value = {
        "id": "mock_id",
        "label": "mock_label"
    }

    result = get_brain_region_dict("SSCX", access_token="test_token")
    assert result["id"] == "mock_id"
    assert result["label"] == "mock_label"
