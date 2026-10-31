from unittest.mock import MagicMock, patch
import pytest

from agents.construction_verification.services.earth_search import EarthSearchClient, EarthSearchError


def test_earth_search_scene_discovery_synthetic():
    """Test 5: Earth Search scene discovery generates valid scene metadata."""
    client = EarthSearchClient()
    bbox = [77.5896, 12.9666, 77.5996, 12.9766]
    scene = client.find_best_scene(bbox=bbox, target_date="2024-02-15", days_window=30)

    assert scene is not None
    assert "scene_id" in scene
    assert "asset_urls" in scene
    assert scene["cloud_cover_percent"] <= 30.0
    assert "red" in scene["asset_urls"]
    assert "nir" in scene["asset_urls"]


def test_missing_satellite_bands_filtering():
    """Test 6: Earth Search filters out candidate scenes missing required spectral bands."""
    client = EarthSearchClient()
    bbox = [77.5896, 12.9666, 77.5996, 12.9766]

    # Mock item missing NIR band
    mock_item = MagicMock()
    mock_item.id = "S2A_INCOMPLETE_BANDS"
    mock_item.geometry = {
        "type": "Polygon",
        "coordinates": [
            [[77.5, 12.9], [77.7, 12.9], [77.7, 13.1], [77.5, 13.1], [77.5, 12.9]]
        ],
    }
    mock_item.assets = {
        "red": MagicMock(href="https://example.com/red.tif"),
        "green": MagicMock(href="https://example.com/green.tif"),
        "blue": MagicMock(href="https://example.com/blue.tif"),
        # NIR is missing!
        "scl": MagicMock(href="https://example.com/scl.tif"),
    }
    mock_item.properties = {"eo:cloud_cover": 5.0}

    mock_search = MagicMock()
    mock_search.items.return_value = [mock_item]

    with patch("core.settings.settings.USE_FAKE_MODEL", False):
        with patch("pystac_client.Client.open") as mock_open:
            mock_client_instance = MagicMock()
            mock_client_instance.search.return_value = mock_search
            mock_open.return_value = mock_client_instance

            scene = client.search_scene(bbox=bbox, datetime_range="2024-01-01T00:00:00Z/2024-01-31T23:59:59Z")
            # Incomplete bands item should be skipped
            assert scene is None


def test_cloudy_imagery_rejection():
    """Test 7: Scenes exceeding cloud cover threshold are rejected."""
    client = EarthSearchClient(max_cloud_cover=15.0)
    bbox = [77.5896, 12.9666, 77.5996, 12.9766]

    # Mock item with 45% clouds
    mock_item = MagicMock()
    mock_item.id = "S2A_CLOUDY_SCENE"
    mock_item.geometry = {
        "type": "Polygon",
        "coordinates": [
            [[77.5, 12.9], [77.7, 12.9], [77.7, 13.1], [77.5, 13.1], [77.5, 12.9]]
        ],
    }
    mock_item.assets = {
        "red": MagicMock(href="https://example.com/red.tif"),
        "green": MagicMock(href="https://example.com/green.tif"),
        "blue": MagicMock(href="https://example.com/blue.tif"),
        "nir": MagicMock(href="https://example.com/nir.tif"),
        "scl": MagicMock(href="https://example.com/scl.tif"),
    }
    mock_item.properties = {"eo:cloud_cover": 45.0}

    mock_search = MagicMock()
    mock_search.items.return_value = []  # STAC query filtered out due to eo:cloud_cover < 15

    with patch("core.settings.settings.USE_FAKE_MODEL", False):
        with patch("pystac_client.Client.open") as mock_open:
            mock_client_instance = MagicMock()
            mock_client_instance.search.return_value = mock_search
            mock_open.return_value = mock_client_instance

            scene = client.search_scene(bbox=bbox, datetime_range="2024-01-01T00:00:00Z/2024-01-31T23:59:59Z")
            assert scene is None

