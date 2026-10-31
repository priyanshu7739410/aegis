import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import rasterio
import rasterio.features
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import from_bounds
from shapely.geometry import shape

from core.settings import settings
from agents.construction_verification.schemas import (
    DiscoveryCandidate,
    DiscoveryReport,
    VisionExecutionAudit,
)
from agents.construction_verification.services.earth_search import EarthSearchClient, EarthSearchError
from agents.construction_verification.services.gemini_client import GeminiVerificationClient

logger = logging.getLogger(__name__)

# Sentinel-2 SCL invalid classes: 0=No Data, 1=Saturated, 3=Cloud Shadow, 8=Cloud Medium, 9=Cloud High, 10=Thin Cirrus
SCL_INVALID_CLASSES = {0, 1, 3, 8, 9, 10}


class DiscoveryError(Exception):
    """Exception raised during independent construction discovery operations."""
    pass


class ConstructionDiscoveryService:
    """Service for independent satellite change detection and construction discovery."""

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir or settings.CONSTRUCTION_OUTPUT_DIR) / "discovery"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.earth_search = EarthSearchClient()
        self.gemini_client = GeminiVerificationClient()

    @staticmethod
    def calculate_bsi(
        swir: np.ndarray, red: np.ndarray, nir: np.ndarray, blue: np.ndarray
    ) -> np.ndarray:
        """Compute Bare Soil Index: ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))."""
        swir_f = swir.astype(np.float32)
        red_f = red.astype(np.float32)
        nir_f = nir.astype(np.float32)
        blue_f = blue.astype(np.float32)

        numerator = (swir_f + red_f) - (nir_f + blue_f)
        denominator = (swir_f + red_f) + (nir_f + blue_f)
        denominator = np.where(denominator == 0, 1e-7, denominator)

        bsi = numerator / denominator
        return np.clip(bsi, -1.0, 1.0)

    @staticmethod
    def calculate_bounding_box(lat: float, lon: float, radius_km: float = 5.0) -> list[float]:
        """Compute WGS84 bounding box [min_lon, min_lat, max_lon, max_lat] for a centroid and radius."""
        delta_lat = radius_km / 111.0
        # Prevent division by zero at poles
        cos_lat = max(np.cos(np.radians(lat)), 1e-4)
        delta_lon = radius_km / (111.0 * cos_lat)
        return [
            round(lon - delta_lon, 6),
            round(lat - delta_lat, 6),
            round(lon + delta_lon, 6),
            round(lat + delta_lat, 6),
        ]

    def run_discovery(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
        before_date_start: str = "2021-01-01",
        before_date_end: str = "2021-03-31",
        after_date_start: str = "2025-01-01",
        after_date_end: str = "2025-03-31",
        min_change_area_m2: float = 5000.0,
        max_cloud_cover: float = 20.0,
        area_name: str = "Independent Site Discovery",
        discovery_id: str | None = None,
    ) -> DiscoveryReport:
        """Execute independent construction discovery without requiring any contractor document or pre-existing claims."""
        disc_id = discovery_id or f"DISC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        run_dir = self.output_dir / disc_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Construction Discovery {disc_id} at ({latitude}, {longitude}), radius={radius_km}km")

        # 1. Validate area & calculate bounding box
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            raise DiscoveryError(f"Invalid coordinates: lat={latitude}, lon={longitude}")

        bbox = self.calculate_bounding_box(latitude, longitude, radius_km)

        # 2. Retrieve before and after satellite scenes via Element84 STAC
        range_before = f"{before_date_start}T00:00:00Z/{before_date_end}T23:59:59Z"
        range_after = f"{after_date_start}T00:00:00Z/{after_date_end}T23:59:59Z"

        before_scene = self.earth_search.search_scene(bbox=bbox, datetime_range=range_before, max_cloud=max_cloud_cover)
        after_scene = self.earth_search.search_scene(bbox=bbox, datetime_range=range_after, max_cloud=max_cloud_cover)

        if not before_scene:
            raise DiscoveryError(
                f"No cloud-free Sentinel-2 scene (<{max_cloud_cover}% cloud) found for before period {before_date_start} to {before_date_end} at ({latitude}, {longitude})"
            )
        if not after_scene:
            raise DiscoveryError(
                f"No cloud-free Sentinel-2 scene (<{max_cloud_cover}% cloud) found for after period {after_date_start} to {after_date_end} at ({latitude}, {longitude})"
            )

        # 3. Preprocess and align multi-spectral bands
        aligned_data = self._read_and_align_rasters(bbox, before_scene, after_scene)

        # 4. Detect candidate changes
        detection_result = self._detect_changes(
            aligned_data=aligned_data,
            bbox_wgs84=bbox,
            min_change_area_m2=min_change_area_m2,
            run_dir=run_dir,
        )

        candidates = detection_result["candidates"]
        total_disturbed_area_m2 = detection_result["total_disturbed_area_m2"]

        # 5. Multimodal Gemini Vision analysis
        scene_metadata = {
            "before_scene_id": before_scene["scene_id"],
            "after_scene_id": after_scene["scene_id"],
            "before_acquisition_date": before_scene["acquisition_date"],
            "after_acquisition_date": after_scene["acquisition_date"],
            "before_cloud_cover_pct": before_scene.get("cloud_cover_percent", 0.0),
            "after_cloud_cover_pct": after_scene.get("cloud_cover_percent", 0.0),
            "crs": aligned_data["crs"],
            "raster_shape": list(aligned_data["shape"]),
            "resolution_m": 10.0,
            "center": {"latitude": latitude, "longitude": longitude},
            "radius_km": radius_km,
            "area_name": area_name,
        }

        vision_res = self.gemini_client.analyze_discovery_imagery(
            rgb_before_path=detection_result["rgb_before_path"],
            rgb_after_path=detection_result["rgb_after_path"],
            candidates=candidates,
            metrics={
                "total_disturbed_area_m2": total_disturbed_area_m2,
                "delta_ndvi_mean": detection_result["delta_ndvi_mean"],
                "candidates_count": len(candidates),
            },
            scene_metadata=scene_metadata,
        )

        vision_summary = vision_res["summary"]
        vision_audit = vision_res["audit"]
        candidate_interpretations = vision_res.get("candidate_interpretations", {})

        # Integrate Gemini interpretations into candidate objects
        candidate_objects: list[DiscoveryCandidate] = []
        for c in candidates:
            c_id = c["candidate_id"]
            interp = candidate_interpretations.get(c_id, f"Candidate surface disturbance with {c['area_hectares']} ha extent.")
            candidate_objects.append(
                DiscoveryCandidate(
                    candidate_id=c_id,
                    centroid=c["centroid"],
                    bbox=c["bbox"],
                    area_m2=c["area_m2"],
                    area_hectares=c["area_hectares"],
                    mean_delta_ndvi=c["mean_delta_ndvi"],
                    mean_delta_bsi=c["mean_delta_bsi"],
                    change_type=c.get("change_type", "Surface Disturbance / Construction Candidate"),
                    geometry=c["geometry"],
                    gemini_interpretation=interp,
                    confidence=c.get("confidence"),
                )
            )

        # 6. Build GeoJSON FeatureCollection
        features = []
        for c in candidate_objects:
            features.append({
                "type": "Feature",
                "id": c.candidate_id,
                "geometry": c.geometry,
                "properties": {
                    "candidate_id": c.candidate_id,
                    "centroid_lat": c.centroid["latitude"],
                    "centroid_lon": c.centroid["longitude"],
                    "area_m2": c.area_m2,
                    "area_hectares": c.area_hectares,
                    "mean_delta_ndvi": c.mean_delta_ndvi,
                    "mean_delta_bsi": c.mean_delta_bsi,
                    "change_type": c.change_type,
                    "gemini_interpretation": c.gemini_interpretation,
                    "provenance_before_scene": before_scene["scene_id"],
                    "provenance_after_scene": after_scene["scene_id"],
                },
            })

        geojson_fc = {
            "type": "FeatureCollection",
            "name": f"Construction_Candidates_{disc_id}",
            "features": features,
        }

        geojson_path = run_dir / "candidates.geojson"
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson_fc, f, indent=2)

        report = DiscoveryReport(
            discovery_id=disc_id,
            area_name=area_name,
            center={"latitude": latitude, "longitude": longitude},
            radius_km=radius_km,
            bounding_box=bbox,
            before_scene_id=before_scene["scene_id"],
            before_acquisition_date=before_scene["acquisition_date"],
            before_cloud_cover_pct=before_scene.get("cloud_cover_percent", 0.0),
            after_scene_id=after_scene["scene_id"],
            after_acquisition_date=after_scene["acquisition_date"],
            after_cloud_cover_pct=after_scene.get("cloud_cover_percent", 0.0),
            crs=aligned_data["crs"],
            raster_shape=list(aligned_data["shape"]),
            resolution_m=10.0,
            total_disturbed_area_m2=total_disturbed_area_m2,
            total_disturbed_area_hectares=round(total_disturbed_area_m2 / 10000.0, 2),
            candidates_count=len(candidate_objects),
            candidates=candidate_objects,
            gemini_vision_summary=vision_summary,
            vision_audit=vision_audit,
            geojson_feature_collection=geojson_fc,
            rgb_before_path=detection_result["rgb_before_path"],
            rgb_after_path=detection_result["rgb_after_path"],
            change_mask_path=detection_result["change_mask_path"],
            ndvi_diff_path=detection_result["ndvi_diff_path"],
        )

        report_path = run_dir / "discovery_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        logger.info(f"Construction Discovery {disc_id} complete. Detected {len(candidate_objects)} candidate polygons.")
        return report

    def _read_and_align_rasters(
        self, bbox_wgs84: list[float], before_scene: dict[str, Any], after_scene: dict[str, Any]
    ) -> dict[str, Any]:
        """Read and align multi-spectral bands onto an identical CRS and pixel grid."""
        b_assets = before_scene.get("asset_urls", {})
        a_assets = after_scene.get("asset_urls", {})

        is_synthetic = (
            settings.USE_FAKE_MODEL
            or any(url.startswith("mock://") for url in b_assets.values())
            or any(url.startswith("mock://") for url in a_assets.values())
        )

        if is_synthetic:
            # Deterministic synthetic grid for tests
            shape_grid = (500, 500)
            np.random.seed(42)
            r1 = np.full(shape_grid, 400.0, dtype=np.float32)
            g1 = np.full(shape_grid, 600.0, dtype=np.float32)
            b1 = np.full(shape_grid, 300.0, dtype=np.float32)
            n1 = np.full(shape_grid, 2500.0, dtype=np.float32)
            scl1 = np.full(shape_grid, 4, dtype=np.uint8)

            r2 = np.full(shape_grid, 1200.0, dtype=np.float32)
            g2 = np.full(shape_grid, 1100.0, dtype=np.float32)
            b2 = np.full(shape_grid, 900.0, dtype=np.float32)
            n2 = np.full(shape_grid, 1800.0, dtype=np.float32)
            scl2 = np.full(shape_grid, 5, dtype=np.uint8)

            # Insert runway-shaped clearing
            r2[200:250, 100:400] = 1600.0
            n2[200:250, 100:400] = 1200.0

            sw1 = np.full(shape_grid, 800.0, dtype=np.float32)
            sw2 = np.full(shape_grid, 2200.0, dtype=np.float32)
            sw2[200:250, 100:400] = 2600.0

            # Derive transform
            dx = (bbox_wgs84[2] - bbox_wgs84[0]) / shape_grid[1]
            dy = (bbox_wgs84[3] - bbox_wgs84[1]) / shape_grid[0]
            win_transform = rasterio.transform.from_origin(bbox_wgs84[0], bbox_wgs84[3], dx, dy)
            crs_str = "EPSG:4326"

            return {
                "r1": r1, "g1": g1, "b1": b1, "n1": n1, "sw1": sw1, "scl1": scl1,
                "r2": r2, "g2": g2, "b2": b2, "n2": n2, "sw2": sw2, "scl2": scl2,
                "shape": shape_grid,
                "crs": crs_str,
                "transform": win_transform,
            }

        def read_asset(url: str, target_shape=None, resampling=Resampling.bilinear):
            with rasterio.open(url) as src:
                nb = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
                win = from_bounds(*nb, transform=src.transform)
                if target_shape:
                    data = src.read(1, window=win, out_shape=target_shape, resampling=resampling, boundless=True)
                else:
                    data = src.read(1, window=win, boundless=True)
                return data.astype(np.float32), src.crs, rasterio.windows.transform(win, src.transform)

        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True):
            r1, crs_obj, win_transform = read_asset(b_assets["red"])
            target_shape = r1.shape

            g1, _, _ = read_asset(b_assets["green"], target_shape=target_shape)
            b1, _, _ = read_asset(b_assets["blue"], target_shape=target_shape)
            n1, _, _ = read_asset(b_assets["nir"], target_shape=target_shape)
            scl1, _, _ = read_asset(b_assets["scl"], target_shape=target_shape, resampling=Resampling.nearest)
            sw1 = None
            b_swir_url = b_assets.get("swir16") or b_assets.get("swir22") or b_assets.get("B11")
            if b_swir_url:
                sw1, _, _ = read_asset(b_swir_url, target_shape=target_shape, resampling=Resampling.bilinear)

            r2, _, _ = read_asset(a_assets["red"], target_shape=target_shape)
            g2, _, _ = read_asset(a_assets["green"], target_shape=target_shape)
            b2, _, _ = read_asset(a_assets["blue"], target_shape=target_shape)
            n2, _, _ = read_asset(a_assets["nir"], target_shape=target_shape)
            scl2, _, _ = read_asset(a_assets["scl"], target_shape=target_shape, resampling=Resampling.nearest)
            sw2 = None
            a_swir_url = a_assets.get("swir16") or a_assets.get("swir22") or a_assets.get("B11")
            if a_swir_url:
                sw2, _, _ = read_asset(a_swir_url, target_shape=target_shape, resampling=Resampling.bilinear)

        return {
            "r1": r1, "g1": g1, "b1": b1, "n1": n1, "sw1": sw1, "scl1": scl1,
            "r2": r2, "g2": g2, "b2": b2, "n2": n2, "sw2": sw2, "scl2": scl2,
            "shape": target_shape,
            "crs": str(crs_obj),
            "transform": win_transform,
        }

    def _detect_changes(
        self,
        aligned_data: dict[str, Any],
        bbox_wgs84: list[float],
        min_change_area_m2: float,
        run_dir: Path,
    ) -> dict[str, Any]:
        """Compute spectral indices, cloud mask, and vectorize candidate construction polygons."""
        r1 = aligned_data["r1"]
        g1 = aligned_data["g1"]
        b1 = aligned_data["b1"]
        n1 = aligned_data["n1"]
        scl1 = aligned_data["scl1"]

        r2 = aligned_data["r2"]
        g2 = aligned_data["g2"]
        b2 = aligned_data["b2"]
        n2 = aligned_data["n2"]
        scl2 = aligned_data["scl2"]

        crs_str = aligned_data["crs"]
        win_transform = aligned_data["transform"]

        # Valid pixel masking
        invalid1 = np.isin(scl1.astype(int), list(SCL_INVALID_CLASSES))
        invalid2 = np.isin(scl2.astype(int), list(SCL_INVALID_CLASSES))
        valid_mask = ~(invalid1 | invalid2)

        # NDVI calculation
        ndvi1 = (n1 - r1) / (n1 + r1 + 1e-7)
        ndvi2 = (n2 - r2) / (n2 + r2 + 1e-7)
        delta_ndvi = ndvi2 - ndvi1

        # Bare Soil Index (BSI) calculation if SWIR is available
        sw1 = aligned_data.get("sw1")
        sw2 = aligned_data.get("sw2")
        delta_bsi = None
        if sw1 is not None and sw2 is not None:
            bsi1 = self.calculate_bsi(sw1, r1, n1, b1)
            bsi2 = self.calculate_bsi(sw2, r2, n2, b2)
            delta_bsi = bsi2 - bsi1

        # Candidate change mask: significant vegetation loss on valid pixels
        change_mask = (delta_ndvi < -0.15) & valid_mask
        disturbed_pixels = int(np.sum(change_mask))
        total_disturbed_area_m2 = float(disturbed_pixels * 100.0)

        # Save RGB previews
        rgb_before_path = run_dir / "rgb_before.png"
        rgb_after_path = run_dir / "rgb_after.png"
        ndvi_diff_path = run_dir / "ndvi_diff.png"
        change_mask_path = run_dir / "change_mask.png"

        self._save_normalized_rgb(r1, g1, b1, rgb_before_path)
        self._save_normalized_rgb(r2, g2, b2, rgb_after_path)
        self._save_ndvi_diff(delta_ndvi, valid_mask, ndvi_diff_path)
        self._save_change_mask(change_mask, change_mask_path)

        # Vectorize contiguous change clusters
        raw_shapes = list(
            rasterio.features.shapes(change_mask.astype(np.uint8), mask=change_mask, transform=win_transform)
        )

        candidates: list[dict[str, Any]] = []
        cand_idx = 1
        for geom_native, val in raw_shapes:
            if val != 1:
                continue
            poly_native = shape(geom_native)
            area_m2 = poly_native.area

            # If CRS is EPSG:4326 (degrees), convert area to m2 approx
            if "4326" in crs_str:
                area_m2 = area_m2 * (111000.0 ** 2) * np.cos(np.radians(bbox_wgs84[1]))

            if area_m2 < min_change_area_m2:
                continue

            # Create a raster mask for this candidate to calculate exact per-candidate statistics
            poly_mask = rasterio.features.geometry_mask(
                [geom_native],
                out_shape=aligned_data["shape"],
                transform=win_transform,
                invert=True,
            )
            poly_valid = poly_mask & valid_mask

            if np.any(poly_valid):
                cand_delta_ndvi = round(float(np.mean(delta_ndvi[poly_valid])), 4)
                cand_delta_bsi = round(float(np.mean(delta_bsi[poly_valid])), 4) if delta_bsi is not None else 0.0
            else:
                cand_delta_ndvi = round(float(np.mean(delta_ndvi[change_mask])), 4)
                cand_delta_bsi = 0.0

            # Transform geometry to EPSG:4326
            if "4326" not in crs_str:
                geom_wgs84 = transform_geom(crs_str, "EPSG:4326", geom_native)
            else:
                geom_wgs84 = geom_native

            poly_wgs84 = shape(geom_wgs84)
            c_lon, c_lat = poly_wgs84.centroid.x, poly_wgs84.centroid.y
            b_bounds = poly_wgs84.bounds

            candidates.append({
                "candidate_id": f"CANDIDATE-{cand_idx:02d}",
                "centroid": {"latitude": round(c_lat, 6), "longitude": round(c_lon, 6)},
                "bbox": [round(b, 6) for b in b_bounds],
                "area_m2": round(area_m2, 1),
                "area_hectares": round(area_m2 / 10000.0, 2),
                "mean_delta_ndvi": cand_delta_ndvi,
                "mean_delta_bsi": cand_delta_bsi,
                "change_type": "Surface Disturbance / Construction Candidate",
                "geometry": geom_wgs84,
            })
            cand_idx += 1

        # Sort largest change areas first, keep top 15
        candidates.sort(key=lambda x: x["area_m2"], reverse=True)
        # Re-index top candidates
        for idx, c in enumerate(candidates):
            c["candidate_id"] = f"CANDIDATE-{idx+1:02d}"

        delta_ndvi_valid = delta_ndvi[valid_mask] if np.any(valid_mask) else delta_ndvi
        mean_delta = float(np.mean(delta_ndvi_valid))

        return {
            "candidates": candidates[:15],
            "total_disturbed_area_m2": total_disturbed_area_m2,
            "delta_ndvi_mean": round(mean_delta, 4),
            "rgb_before_path": str(rgb_before_path),
            "rgb_after_path": str(rgb_after_path),
            "ndvi_diff_path": str(ndvi_diff_path),
            "change_mask_path": str(change_mask_path),
        }

    def _save_normalized_rgb(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, path: Path) -> None:
        """Save contrast-normalized true color RGB image."""
        def norm(band: np.ndarray) -> np.ndarray:
            p2, p98 = np.percentile(band, (2, 98))
            diff = max(p98 - p2, 1e-4)
            return (np.clip((band - p2) / diff, 0.0, 1.0) * 255).astype(np.uint8)

        rgb = np.stack([norm(r), norm(g), norm(b)], axis=-1)
        Image.fromarray(rgb).save(path)

    def _save_ndvi_diff(self, delta_ndvi: np.ndarray, valid_mask: np.ndarray, path: Path) -> None:
        """Save RGB visualization of NDVI change."""
        h, w = delta_ndvi.shape
        rgb = np.full((h, w, 3), 180, dtype=np.uint8)  # Neutral tan/gray

        loss_mask = (delta_ndvi < -0.15) & valid_mask
        growth_mask = (delta_ndvi > 0.15) & valid_mask

        rgb[loss_mask] = [220, 50, 47]     # Red: clearing/loss
        rgb[growth_mask] = [46, 204, 113]  # Green: growth

        Image.fromarray(rgb).save(path)

    def _save_change_mask(self, change_mask: np.ndarray, path: Path) -> None:
        """Save binary change mask PNG."""
        h, w = change_mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[change_mask] = [255, 215, 0]  # Amber/Yellow for candidate change
        Image.fromarray(rgb).save(path)

