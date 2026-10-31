import numpy as np
import pytest

from agents.construction_verification.services.raster_processing import RasterProcessor


def test_ndvi_calculation_accuracy():
    """Test 8: NDVI calculation formula: (NIR - Red) / (NIR + Red)."""
    nir = np.array([2000.0, 500.0, 0.0], dtype=np.float32)
    red = np.array([400.0, 500.0, 0.0], dtype=np.float32)

    ndvi = RasterProcessor.calculate_ndvi(nir, red)

    # Pixel 1: (2000 - 400) / (2000 + 400) = 1600 / 2400 = 0.6667 (Dense vegetation)
    assert pytest.approx(ndvi[0], 0.001) == 0.6667
    # Pixel 2: (500 - 500) / 1000 = 0.0 (Bare ground)
    assert pytest.approx(ndvi[1], 0.001) == 0.0
    # Pixel 3: Division by zero protection
    assert not np.isnan(ndvi[2])
    assert not np.isinf(ndvi[2])


def test_bsi_calculation():
    """Test 8b: Bare Soil Index calculation."""
    swir = np.array([2000.0], dtype=np.float32)
    red = np.array([1500.0], dtype=np.float32)
    nir = np.array([1200.0], dtype=np.float32)
    blue = np.array([800.0], dtype=np.float32)

    # BSI = ((2000 + 1500) - (1200 + 800)) / ((2000 + 1500) + (1200 + 800))
    # = (3500 - 2000) / (3500 + 2000) = 1500 / 5500 = 0.2727
    bsi = RasterProcessor.calculate_bsi(swir, red, nir, blue)
    assert pytest.approx(bsi[0], 0.001) == 0.2727


def test_raster_alignment_and_window_read(tmp_path):
    """Test 9: Windowed raster read returns consistent target grid shape."""
    processor = RasterProcessor(output_dir=tmp_path)
    target_shape = (50, 50)
    arr, _ = processor.read_window_band(
        asset_url="mock://nir.tif",
        bbox_wgs84=[77.5, 12.9, 77.6, 13.0],
        target_shape=target_shape,
    )
    assert arr.shape == target_shape
    assert arr.dtype == np.float32


def test_vegetation_loss_and_disturbed_area(tmp_path):
    """Test 10: Vegetation loss fraction and disturbed area calculation."""
    processor = RasterProcessor(output_dir=tmp_path)

    before_scene = {
        "scene_id": "S2A_BEFORE",
        "acquisition_date": "2024-01-15",
        "asset_urls": {"nir": "mock://nir", "red": "mock://red"},
    }
    after_scene = {
        "scene_id": "S2A_AFTER",
        "acquisition_date": "2024-04-15",
        "asset_urls": {"nir": "mock://nir", "red": "mock://red"},
    }

    metrics = processor.analyze_change(
        case_id="TEST-CASE",
        bbox_wgs84=[77.58, 12.96, 77.60, 12.98],
        before_scene=before_scene,
        after_scene=after_scene,
    )

    assert "mean_ndvi_before" in metrics
    assert "mean_ndvi_after" in metrics
    assert "delta_ndvi" in metrics
    assert "vegetation_loss_fraction" in metrics
    assert "potential_disturbed_area_m2" in metrics
    assert metrics["mean_ndvi_before"] > metrics["mean_ndvi_after"]
    assert metrics["delta_ndvi"] < 0
    assert metrics["vegetation_loss_fraction"] > 0.5
    assert metrics["potential_disturbed_area_m2"] > 0

