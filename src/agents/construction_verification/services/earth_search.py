import logging
from datetime import datetime, timedelta
from typing import Any
import pystac_client
from shapely.geometry import box, shape

from core.settings import settings

logger = logging.getLogger(__name__)

REQUIRED_ASSETS = ["red", "green", "blue", "nir", "scl"]


class EarthSearchError(Exception):
    """Exception raised for errors during Earth Search STAC operations."""
    pass


class EarthSearchClient:
    """Client for querying Sentinel-2 L2A STAC items via Earth Search AWS."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        collection: str | None = None,
        max_cloud_cover: float | None = None,
    ):
        self.endpoint_url = endpoint_url or settings.EARTH_SEARCH_URL
        self.collection = collection or settings.EARTH_SEARCH_COLLECTION
        self.max_cloud_cover = (
            max_cloud_cover if max_cloud_cover is not None else settings.MAX_CLOUD_COVER
        )

    def find_best_scene(
        self,
        bbox: list[float],
        target_date: str,
        days_window: int = 60,
        max_cloud: float | None = None,
        prefer_before: bool = False,
    ) -> dict[str, Any] | None:
        """Find the best cloud-free Sentinel-2 scene around a target date.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84
            target_date: Date string 'YYYY-MM-DD'
            days_window: Search window in days
            max_cloud: Max allowed scene cloud cover percentage
            prefer_before: If True, searches backward from target date; if False, searches forward or centered.
        """
        cloud_thresh = max_cloud if max_cloud is not None else self.max_cloud_cover

        # Calculate search date window
        try:
            target_dt = datetime.strptime(target_date[:10], "%Y-%m-%d")
        except Exception as e:
            raise EarthSearchError(f"Invalid target date format '{target_date}': {e}")

        if prefer_before:
            start_dt = target_dt - timedelta(days=days_window)
            end_dt = target_dt
        else:
            start_dt = target_dt
            end_dt = target_dt + timedelta(days=days_window)

        date_range_str = f"{start_dt.strftime('%Y-%m-%d')}T00:00:00Z/{end_dt.strftime('%Y-%m-%d')}T23:59:59Z"

        return self.search_scene(bbox=bbox, datetime_range=date_range_str, max_cloud=cloud_thresh)

    def search_scenes(
        self,
        bbox: list[float],
        datetime_range: str,
        max_cloud: float = 30.0,
        max_items: int = 15,
    ) -> list[dict[str, Any]]:
        """Query STAC API for Sentinel-2 scenes meeting spatial, temporal, and cloud criteria."""
        if settings.USE_FAKE_MODEL:
            logger.info("Using synthetic Earth Search scene (USE_FAKE_MODEL=True)")
            return [self._get_synthetic_scene(bbox, datetime_range)]

        try:
            client = pystac_client.Client.open(self.endpoint_url)
            search = client.search(
                collections=[self.collection],
                bbox=bbox,
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": max_cloud}},
                sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
                max_items=max_items,
            )

            items = list(search.items())
            if not items:
                logger.warning(
                    f"No Sentinel-2 scenes found in collection {self.collection} for bbox {bbox} "
                    f"in range {datetime_range} with cloud < {max_cloud}%"
                )
                return []

            matched_scenes: list[dict[str, Any]] = []
            bbox_geom = box(*bbox)
            for item in items:
                # Check geometry intersection
                item_geom = shape(item.geometry)
                if not item_geom.intersects(bbox_geom):
                    continue

                # Validate required assets
                asset_keys = set(item.assets.keys())
                has_red = "red" in asset_keys or "B04" in asset_keys
                has_green = "green" in asset_keys or "B03" in asset_keys
                has_blue = "blue" in asset_keys or "B02" in asset_keys
                has_nir = "nir" in asset_keys or "nir08" in asset_keys or "B08" in asset_keys
                has_scl = "scl" in asset_keys or "SCL" in asset_keys

                if not (has_red and has_green and has_blue and has_nir and has_scl):
                    logger.debug(f"Item {item.id} skipped due to missing required spectral bands.")
                    continue

                asset_urls: dict[str, str] = {}
                for key in ["red", "green", "blue", "nir", "nir08", "scl", "swir16", "swir22", "visual"]:
                    if key in item.assets:
                        asset_urls[key] = item.assets[key].href

                # Normalize asset keys
                if "nir" not in asset_urls and "nir08" in asset_urls:
                    asset_urls["nir"] = asset_urls["nir08"]

                matched_scenes.append({
                    "scene_id": item.id,
                    "acquisition_date": item.datetime.strftime("%Y-%m-%d") if item.datetime else datetime_range[:10],
                    "cloud_cover_percent": float(item.properties.get("eo:cloud_cover", 0.0)),
                    "asset_urls": asset_urls,
                    "bbox": item.bbox,
                    "properties": item.properties,
                })

            return matched_scenes

        except Exception as e:
            logger.error(f"Error querying Earth Search STAC: {e}")
            if settings.USE_FAKE_MODEL:
                return [self._get_synthetic_scene(bbox, datetime_range)]
            raise EarthSearchError(f"Failed querying STAC service: {e}") from e

    def search_scene(
        self,
        bbox: list[float],
        datetime_range: str,
        max_cloud: float = 30.0,
    ) -> dict[str, Any] | None:
        """Query STAC API for the single best Sentinel-2 scene meeting criteria."""
        scenes = self.search_scenes(bbox=bbox, datetime_range=datetime_range, max_cloud=max_cloud)
        return scenes[0] if scenes else None

    def _get_synthetic_scene(self, bbox: list[float], datetime_range: str) -> dict[str, Any]:
        """Generate deterministic synthetic scene for offline testing/verification."""
        date_str = datetime_range.split("/")[0][:10] if "/" in datetime_range else "2024-01-15"
        return {
            "scene_id": f"S2A_MSIL2A_{date_str.replace('-', '')}_SYNTHETIC",
            "acquisition_date": date_str,
            "cloud_cover_percent": 4.5,
            "asset_urls": {
                "red": "mock://red.tif",
                "green": "mock://green.tif",
                "blue": "mock://blue.tif",
                "nir": "mock://nir.tif",
                "scl": "mock://scl.tif",
                "swir16": "mock://swir16.tif",
            },
            "bbox": bbox,
            "properties": {"eo:cloud_cover": 4.5, "synthetic": True},
        }

