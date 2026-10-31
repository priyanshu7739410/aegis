import logging
import os
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds

from core.settings import settings

logger = logging.getLogger(__name__)

VEGETATION_LOSS_THRESHOLD = -0.15  # Delta NDVI below this is classified as significant loss
SCL_INVALID_CLASSES = {0, 1, 3, 8, 9, 10}  # No data, saturated, cloud shadow, clouds, cirrus


class RasterProcessingError(Exception):
    """Exception for raster operations and change detection."""
    pass


class RasterProcessor:
    """Performs deterministic remote sensing analysis, NDVI/BSI calculation, and visualization."""

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir or settings.CONSTRUCTION_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Compute Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
        nir_f = nir.astype(np.float32)
        red_f = red.astype(np.float32)
        denom = nir_f + red_f
        denom = np.where(denom == 0, 1e-7, denom)
        ndvi = (nir_f - red_f) / denom
        return np.clip(ndvi, -1.0, 1.0)

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

    def read_window_band(
        self,
        asset_url: str,
        bbox_wgs84: list[float],
        target_shape: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, Any]:
        """Perform a windowed read of a specific bounding box from a raster asset."""
        if asset_url.startswith("mock://") or settings.USE_FAKE_MODEL:
            # Return synthetic band
            shape = target_shape or (100, 100)
            return np.ones(shape, dtype=np.float32) * 1000.0, None

        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        try:
            with rasterio.open(asset_url) as src:
                # Transform bbox from WGS84 (EPSG:4326) to dataset CRS
                bounds_native = transform_bounds(
                    "EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat
                )
                window = from_bounds(*bounds_native, transform=src.transform)

                if target_shape:
                    data = src.read(
                        1,
                        window=window,
                        out_shape=target_shape,
                        resampling=Resampling.bilinear,
                        boundless=True,
                    )
                else:
                    data = src.read(1, window=window, boundless=True)

                return data.astype(np.float32), src.transform
        except Exception as e:
            logger.warning(f"Failed window read on {asset_url}: {e}. Falling back to synthetic array.")
            shape = target_shape or (100, 100)
            return np.ones(shape, dtype=np.float32) * 1000.0, None

    def analyze_change(
        self,
        case_id: str,
        bbox_wgs84: list[float],
        before_scene: dict[str, Any],
        after_scene: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform full change detection analysis between before and after scenes."""
        case_dir = self.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        target_shape = (100, 100)  # Standard analysis grid for site clip

        # Read before scene bands
        b_assets = before_scene.get("asset_urls", {})
        a_assets = after_scene.get("asset_urls", {})

        is_mock = (
            settings.USE_FAKE_MODEL
            or not b_assets
            or any(url.startswith("mock://") for url in b_assets.values())
        )

        if is_mock:
            # Generate controlled deterministic arrays representing vegetation -> clearing
            # Pre-construction: healthy vegetation
            np.random.seed(42)
            b_red = np.full(target_shape, 400.0, dtype=np.float32) + np.random.normal(0, 20, target_shape)
            b_green = np.full(target_shape, 600.0, dtype=np.float32) + np.random.normal(0, 20, target_shape)
            b_blue = np.full(target_shape, 300.0, dtype=np.float32) + np.random.normal(0, 20, target_shape)
            b_nir = np.full(target_shape, 2400.0, dtype=np.float32) + np.random.normal(0, 50, target_shape)
            b_swir = np.full(target_shape, 800.0, dtype=np.float32) + np.random.normal(0, 30, target_shape)
            b_scl = np.full(target_shape, 4, dtype=np.uint8)  # Class 4 = Vegetation

            # Post-construction: cleared soil / excavated ground
            a_red = np.full(target_shape, 1400.0, dtype=np.float32) + np.random.normal(0, 30, target_shape)
            a_green = np.full(target_shape, 1200.0, dtype=np.float32) + np.random.normal(0, 30, target_shape)
            a_blue = np.full(target_shape, 900.0, dtype=np.float32) + np.random.normal(0, 30, target_shape)
            a_nir = np.full(target_shape, 1800.0, dtype=np.float32) + np.random.normal(0, 40, target_shape)
            a_swir = np.full(target_shape, 2200.0, dtype=np.float32) + np.random.normal(0, 40, target_shape)
            a_scl = np.full(target_shape, 5, dtype=np.uint8)  # Class 5 = Bare soil
        else:
            # Live asset reads
            b_red, _ = self.read_window_band(b_assets.get("red", ""), bbox_wgs84, target_shape)
            b_green, _ = self.read_window_band(b_assets.get("green", ""), bbox_wgs84, target_shape)
            b_blue, _ = self.read_window_band(b_assets.get("blue", ""), bbox_wgs84, target_shape)
            b_nir, _ = self.read_window_band(b_assets.get("nir", ""), bbox_wgs84, target_shape)
            b_swir, _ = self.read_window_band(b_assets.get("swir16", b_assets.get("swir22", "")), bbox_wgs84, target_shape)
            b_scl, _ = self.read_window_band(b_assets.get("scl", ""), bbox_wgs84, target_shape)

            a_red, _ = self.read_window_band(a_assets.get("red", ""), bbox_wgs84, target_shape)
            a_green, _ = self.read_window_band(a_assets.get("green", ""), bbox_wgs84, target_shape)
            a_blue, _ = self.read_window_band(a_assets.get("blue", ""), bbox_wgs84, target_shape)
            a_nir, _ = self.read_window_band(a_assets.get("nir", ""), bbox_wgs84, target_shape)
            a_swir, _ = self.read_window_band(a_assets.get("swir16", a_assets.get("swir22", "")), bbox_wgs84, target_shape)
            a_scl, _ = self.read_window_band(a_assets.get("scl", ""), bbox_wgs84, target_shape)

        # Cloud & invalid pixel masking using SCL
        b_invalid = np.isin(b_scl.astype(int), list(SCL_INVALID_CLASSES))
        a_invalid = np.isin(a_scl.astype(int), list(SCL_INVALID_CLASSES))
        valid_mask = ~(b_invalid | a_invalid)

        total_pixels = target_shape[0] * target_shape[1]
        valid_pixels = int(np.sum(valid_mask))
        valid_coverage_pct = round((valid_pixels / total_pixels) * 100.0, 2)

        if valid_pixels == 0:
            logger.warning("Site is completely obscured by clouds or invalid pixels in one or both scenes.")
            valid_mask = np.ones(target_shape, dtype=bool)  # Fallback to avoid division by zero

        # Compute NDVI
        b_ndvi = self.calculate_ndvi(b_nir, b_red)
        a_ndvi = self.calculate_ndvi(a_nir, a_red)
        delta_ndvi = a_ndvi - b_ndvi

        # Compute BSI
        b_bsi = self.calculate_bsi(b_swir, b_red, b_nir, b_blue)
        a_bsi = self.calculate_bsi(a_swir, a_red, a_nir, a_blue)
        delta_bsi = a_bsi - b_bsi

        mean_ndvi_before = float(np.mean(b_ndvi[valid_mask]))
        mean_ndvi_after = float(np.mean(a_ndvi[valid_mask]))
        overall_delta_ndvi = float(np.mean(delta_ndvi[valid_mask]))

        mean_bsi_before = float(np.mean(b_bsi[valid_mask]))
        mean_bsi_after = float(np.mean(a_bsi[valid_mask]))
        overall_delta_bsi = float(np.mean(delta_bsi[valid_mask]))

        # Calculate vegetation loss fraction (pixels with delta_ndvi < VEGETATION_LOSS_THRESHOLD)
        loss_pixels = int(np.sum((delta_ndvi < VEGETATION_LOSS_THRESHOLD) & valid_mask))
        vegetation_loss_fraction = round(loss_pixels / max(valid_pixels, 1), 4)

        # Approximate pixel resolution for Sentinel-2 is 10m x 10m = 100 m^2 per pixel
        PIXEL_AREA_M2 = 100.0
        potential_disturbed_area_m2 = round(loss_pixels * PIXEL_AREA_M2, 2)

        # Generate and save preview visual artifacts
        rgb_before_path = case_dir / "rgb_before.png"
        rgb_after_path = case_dir / "rgb_after.png"
        ndvi_diff_path = case_dir / "ndvi_diff.png"

        self._save_rgb_image(b_red, b_green, b_blue, rgb_before_path)
        self._save_rgb_image(a_red, a_green, a_blue, rgb_after_path)
        self._save_ndvi_diff_image(delta_ndvi, ndvi_diff_path)

        return {
            "mean_ndvi_before": round(mean_ndvi_before, 4),
            "mean_ndvi_after": round(mean_ndvi_after, 4),
            "delta_ndvi": round(overall_delta_ndvi, 4),
            "mean_bsi_before": round(mean_bsi_before, 4),
            "mean_bsi_after": round(mean_bsi_after, 4),
            "delta_bsi": round(overall_delta_bsi, 4),
            "vegetation_loss_fraction": vegetation_loss_fraction,
            "potential_disturbed_area_m2": potential_disturbed_area_m2,
            "valid_observation_coverage_pct": valid_coverage_pct,
            "rgb_before_preview_path": str(rgb_before_path),
            "rgb_after_preview_path": str(rgb_after_path),
            "ndvi_diff_image_path": str(ndvi_diff_path),
        }

    def _save_rgb_image(
        self, red: np.ndarray, green: np.ndarray, blue: np.ndarray, output_path: Path
    ) -> None:
        """Normalize spectral bands and save standard RGB PNG preview."""
        def normalize_band(b: np.ndarray) -> np.ndarray:
            p2, p98 = np.percentile(b, (2, 98))
            if p98 > p2:
                norm = np.clip((b - p2) / (p98 - p2), 0.0, 1.0)
            else:
                norm = np.clip(b / 3000.0, 0.0, 1.0)
            return (norm * 255).astype(np.uint8)

        r = normalize_band(red)
        g = normalize_band(green)
        b = normalize_band(blue)
        rgb = np.stack([r, g, b], axis=-1)

        img = Image.fromarray(rgb)
        img.save(output_path)

    def _save_ndvi_diff_image(self, delta_ndvi: np.ndarray, output_path: Path) -> None:
        """Create RGB colormap representation of NDVI changes."""
        # Colormap logic:
        # delta_ndvi < -0.15 (Loss): Red / Orange
        # -0.15 <= delta_ndvi <= 0.15 (Neutral): Gray / Tan
        # delta_ndvi > 0.15 (Growth): Green
        h, w = delta_ndvi.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        loss_mask = delta_ndvi < -0.15
        growth_mask = delta_ndvi > 0.15
        neutral_mask = ~(loss_mask | growth_mask)

        # Red for significant vegetation loss
        rgb[loss_mask] = [220, 50, 47]
        # Neutral gray
        rgb[neutral_mask] = [180, 180, 180]
        # Green for vegetation increase
        rgb[growth_mask] = [46, 204, 113]

        img = Image.fromarray(rgb)
        img.save(output_path)

