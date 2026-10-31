import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from core.settings import settings

logger = logging.getLogger(__name__)

VEGETATION_LOSS_THRESHOLD = -0.15  # Delta NDVI below this is classified as significant loss
SCL_INVALID_CLASSES = {0, 1, 3, 8, 9, 10}  # No data, saturated, cloud shadow, clouds, cirrus


RESOLUTION_M = 10.0
REQUIRED_BANDS = ("red", "green", "blue", "nir", "scl")


class RasterProcessingError(Exception):
    """Exception for raster operations and change detection."""
    pass


@dataclass(frozen=True)
class AnalysisGrid:
    """Shared pixel grid every band of both scenes is warped onto."""

    crs: str
    transform: Any
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @classmethod
    def from_bbox(cls, bbox_wgs84: list[float], resolution_m: float = RESOLUTION_M) -> "AnalysisGrid":
        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        zone = min(int((center_lon + 180) // 6) + 1, 60)
        crs = f"EPSG:{(32600 if center_lat >= 0 else 32700) + zone}"
        left, bottom, right, top = transform_bounds("EPSG:4326", crs, min_lon, min_lat, max_lon, max_lat)
        width = max(1, math.ceil((right - left) / resolution_m))
        height = max(1, math.ceil((top - bottom) / resolution_m))
        return cls(crs, from_origin(left, top, resolution_m, resolution_m), width, height)


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
        grid: AnalysisGrid | None = None,
        resampling: Resampling = Resampling.bilinear,
    ) -> tuple[np.ndarray, Any]:
        """Warp one raster band onto the analysis grid so scenes in any UTM zone line up."""
        if asset_url.startswith("mock://"):
            if not settings.USE_FAKE_MODEL:
                raise RasterProcessingError(f"Refusing synthetic asset outside test mode: {asset_url}")
            return np.ones(target_shape or (100, 100), dtype=np.float32) * 1000.0, None

        grid = grid or AnalysisGrid.from_bbox(bbox_wgs84)
        try:
            with rasterio.Env(
                AWS_NO_SIGN_REQUEST="YES",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_TIMEOUT="60",
                VSI_CACHE=True,
            ):
                with rasterio.open(asset_url) as src:
                    with WarpedVRT(
                        src,
                        crs=grid.crs,
                        transform=grid.transform,
                        width=grid.width,
                        height=grid.height,
                        resampling=resampling,
                    ) as vrt:
                        return vrt.read(1).astype(np.float32), grid.transform
        except Exception as e:
            logger.error(f"Failed window read on {asset_url}: {e}")
            raise RasterProcessingError(f"Failed to read satellite band from {asset_url}: {e}") from e

    def _read_scene_bands(
        self, scene: dict[str, Any], bbox_wgs84: list[float], grid: AnalysisGrid
    ) -> dict[str, np.ndarray]:
        assets = scene.get("asset_urls", {})
        missing = [band for band in REQUIRED_BANDS if not assets.get(band)]
        if missing:
            raise RasterProcessingError(
                f"Scene {scene.get('scene_id')} is missing required bands: {', '.join(missing)}"
            )

        bands = {
            band: self.read_window_band(assets[band], bbox_wgs84, grid=grid)[0]
            for band in ("red", "green", "blue", "nir")
        }
        # SCL values are class labels, so interpolating them would invent classes.
        bands["scl"] = self.read_window_band(
            assets["scl"], bbox_wgs84, grid=grid, resampling=Resampling.nearest
        )[0]
        swir_url = assets.get("swir16") or assets.get("swir22")
        if swir_url:
            bands["swir"] = self.read_window_band(swir_url, bbox_wgs84, grid=grid)[0]
        return bands

    @staticmethod
    def _synthetic_scene_pair(shape: tuple[int, int]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Test-mode arrays: vegetation before, bare soil after. Never genuine evidence."""
        rng = np.random.default_rng(42)

        def band(mean: float, spread: float) -> np.ndarray:
            return (np.full(shape, mean) + rng.normal(0, spread, shape)).astype(np.float32)

        before = {
            "red": band(400, 20), "green": band(600, 20), "blue": band(300, 20),
            "nir": band(2400, 50), "swir": band(800, 30), "scl": np.full(shape, 4, dtype=np.uint8),
        }
        after = {
            "red": band(1400, 30), "green": band(1200, 30), "blue": band(900, 30),
            "nir": band(1800, 40), "swir": band(2200, 40), "scl": np.full(shape, 5, dtype=np.uint8),
        }
        return before, after

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

        synthetic = settings.USE_FAKE_MODEL
        if not synthetic:
            for scene in (before_scene, after_scene):
                urls = scene.get("asset_urls", {}).values()
                if scene.get("properties", {}).get("synthetic") or any(
                    str(url).startswith("mock://") for url in urls
                ):
                    raise RasterProcessingError(
                        f"Refusing synthetic scene {scene.get('scene_id')} outside test mode."
                    )

        grid = AnalysisGrid.from_bbox(bbox_wgs84)
        target_shape = grid.shape

        if synthetic:
            before, after = self._synthetic_scene_pair(target_shape)
        else:
            before = self._read_scene_bands(before_scene, bbox_wgs84, grid)
            after = self._read_scene_bands(after_scene, bbox_wgs84, grid)

        b_red, b_green, b_blue, b_nir, b_scl = (before[k] for k in ("red", "green", "blue", "nir", "scl"))
        a_red, a_green, a_blue, a_nir, a_scl = (after[k] for k in ("red", "green", "blue", "nir", "scl"))

        # Cloud & invalid pixel masking using SCL
        b_invalid = np.isin(b_scl.astype(int), list(SCL_INVALID_CLASSES))
        a_invalid = np.isin(a_scl.astype(int), list(SCL_INVALID_CLASSES))
        valid_mask = ~(b_invalid | a_invalid)

        total_pixels = target_shape[0] * target_shape[1]
        valid_pixels = int(np.sum(valid_mask))
        valid_coverage_pct = round((valid_pixels / total_pixels) * 100.0, 2)

        if valid_pixels == 0:
            raise RasterProcessingError(
                "Site is completely obscured by clouds or invalid pixels in Sentinel-2 imagery; no valid cloud-free pixels available."
            )

        # Compute NDVI
        b_ndvi = self.calculate_ndvi(b_nir, b_red)
        a_ndvi = self.calculate_ndvi(a_nir, a_red)
        delta_ndvi = a_ndvi - b_ndvi

        mean_ndvi_before = float(np.mean(b_ndvi[valid_mask]))
        mean_ndvi_after = float(np.mean(a_ndvi[valid_mask]))
        overall_delta_ndvi = float(np.mean(delta_ndvi[valid_mask]))

        # BSI needs SWIR, which a scene may not provide; report it as absent rather than guess.
        mean_bsi_before = mean_bsi_after = overall_delta_bsi = None
        if "swir" in before and "swir" in after:
            b_bsi = self.calculate_bsi(before["swir"], b_red, b_nir, b_blue)
            a_bsi = self.calculate_bsi(after["swir"], a_red, a_nir, a_blue)
            mean_bsi_before = round(float(np.mean(b_bsi[valid_mask])), 4)
            mean_bsi_after = round(float(np.mean(a_bsi[valid_mask])), 4)
            overall_delta_bsi = round(float(np.mean((a_bsi - b_bsi)[valid_mask])), 4)

        # Calculate vegetation loss fraction (pixels with delta_ndvi < VEGETATION_LOSS_THRESHOLD)
        loss_pixels = int(np.sum((delta_ndvi < VEGETATION_LOSS_THRESHOLD) & valid_mask))
        vegetation_loss_fraction = round(loss_pixels / max(valid_pixels, 1), 4)

        pixel_area_m2 = RESOLUTION_M * RESOLUTION_M
        potential_disturbed_area_m2 = round(loss_pixels * pixel_area_m2, 2)

        # Generate and save preview visual artifacts
        rgb_before_path = case_dir / "rgb_before.png"
        rgb_after_path = case_dir / "rgb_after.png"
        ndvi_diff_path = case_dir / "ndvi_diff.png"

        # A shared stretch keeps brightness differences between the two dates visible.
        limits = [
            self._stretch_limits(b, a)
            for b, a in ((b_red, a_red), (b_green, a_green), (b_blue, a_blue))
        ]
        self._save_rgb_image(b_red, b_green, b_blue, rgb_before_path, limits)
        self._save_rgb_image(a_red, a_green, a_blue, rgb_after_path, limits)
        self._save_ndvi_diff_image(delta_ndvi, ndvi_diff_path)

        return {
            "mean_ndvi_before": round(mean_ndvi_before, 4),
            "mean_ndvi_after": round(mean_ndvi_after, 4),
            "delta_ndvi": round(overall_delta_ndvi, 4),
            "mean_bsi_before": mean_bsi_before,
            "mean_bsi_after": mean_bsi_after,
            "delta_bsi": overall_delta_bsi,
            "vegetation_loss_fraction": vegetation_loss_fraction,
            "potential_disturbed_area_m2": potential_disturbed_area_m2,
            "valid_observation_coverage_pct": valid_coverage_pct,
            "rgb_before_preview_path": str(rgb_before_path),
            "rgb_after_preview_path": str(rgb_after_path),
            "ndvi_diff_image_path": str(ndvi_diff_path),
            "before_scene_id": before_scene.get("scene_id"),
            "after_scene_id": after_scene.get("scene_id"),
            "crs": grid.crs,
            "raster_shape": list(target_shape),
            "pixel_area_m2": pixel_area_m2,
            "synthetic": synthetic,
        }

    @staticmethod
    def _stretch_limits(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
        p2, p98 = np.percentile(np.concatenate([before.ravel(), after.ravel()]), (2, 98))
        return float(p2), float(p98)

    def _save_rgb_image(
        self,
        red: np.ndarray,
        green: np.ndarray,
        blue: np.ndarray,
        output_path: Path,
        limits: list[tuple[float, float]],
    ) -> None:
        """Stretch spectral bands to the given limits and save an RGB PNG preview."""
        def normalize_band(band: np.ndarray, low: float, high: float) -> np.ndarray:
            span = high - low if high > low else 1.0
            return (np.clip((band - low) / span, 0.0, 1.0) * 255).astype(np.uint8)

        channels = [normalize_band(b, *lim) for b, lim in zip((red, green, blue), limits)]
        Image.fromarray(np.stack(channels, axis=-1)).save(output_path)

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

