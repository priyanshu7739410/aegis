import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
import rasterio

from agents.construction_verification.schemas import (
    CandidateClassification,
    DiscoveryCandidate,
    DiscoveryCandidateFinding,
    DiscoveryReport,
    DiscoveryVisionStructuredOutput,
    VisionExecutionAudit,
)
from agents.construction_verification.services.discovery_service import (
    ConstructionDiscoveryService,
    DiscoveryError,
    SCL_INVALID_CLASSES,
)
from agents.construction_verification.services.gemini_client import GeminiVerificationClient


def test_discovery_does_not_reuse_contractor_case(tmp_path):
    """Test 1: Construction Discovery is independent and does not require or reuse contractor claims or case state."""
    service = ConstructionDiscoveryService(output_dir=tmp_path)

    # Mock EarthSearch to return deterministic synthetic scenes
    mock_b_scene = {
        "scene_id": "S2A_TEST_20210315",
        "acquisition_date": "2021-03-15",
        "cloud_cover_percent": 1.2,
        "asset_urls": {"red": "mock://red.tif", "green": "mock://green.tif", "blue": "mock://blue.tif", "nir": "mock://nir.tif", "scl": "mock://scl.tif"},
    }
    mock_a_scene = {
        "scene_id": "S2A_TEST_20250315",
        "acquisition_date": "2025-03-15",
        "cloud_cover_percent": 0.5,
        "asset_urls": {"red": "mock://red.tif", "green": "mock://green.tif", "blue": "mock://blue.tif", "nir": "mock://nir.tif", "scl": "mock://scl.tif"},
    }

    with patch.object(service.earth_search, "search_scene", side_effect=[mock_b_scene, mock_a_scene]):
        report = service.run_discovery(
            latitude=28.17,
            longitude=77.61,
            radius_km=5.0,
            before_date_start="2021-01-01",
            before_date_end="2021-03-31",
            after_date_start="2025-01-01",
            after_date_end="2025-03-31",
        )

    # Prove discovery produces a fresh discovery report without contractor claim fields
    assert isinstance(report, DiscoveryReport)
    assert report.discovery_id.startswith("DISC-")
    assert not hasattr(report, "construction_claims")
    assert not hasattr(report, "cv_case_id")
    assert report.candidates_count >= 1
    assert report.before_scene_id == "S2A_TEST_20210315"
    assert report.after_scene_id == "S2A_TEST_20250315"


def test_gemini_receives_two_real_image_parts(tmp_path):
    """Test 2: Multimodal Gemini Vision receives two real image base64 data parts, not only text."""
    client = GeminiVerificationClient(api_key="mock_key")

    # Create two dummy PNG files
    im1_path = tmp_path / "before.png"
    im2_path = tmp_path / "after.png"
    Image.new("RGB", (64, 64), color="blue").save(im1_path)
    Image.new("RGB", (64, 64), color="green").save(im2_path)

    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = "Observed runway and clearing features."
    mock_llm.invoke.return_value = mock_resp

    with patch.object(client, "_get_llm", return_value=mock_llm):
        result = client.analyze_discovery_imagery(
            rgb_before_path=str(im1_path),
            rgb_after_path=str(im2_path),
            candidates=[{"candidate_id": "CANDIDATE-01", "centroid": {"latitude": 28.17, "longitude": 77.61}, "area_hectares": 12.5, "area_m2": 125000.0}],
            metrics={"total_disturbed_area_m2": 125000.0, "delta_ndvi_mean": -0.25},
            scene_metadata={"before_scene_id": "S2_B", "after_scene_id": "S2_A", "before_acquisition_date": "2021-03-01", "after_acquisition_date": "2025-03-01", "area_name": "Test"},
        )

    # Verify invoke was called with messages containing two image parts
    assert mock_llm.invoke.called
    messages_passed = mock_llm.invoke.call_args[0][0]
    human_msg = messages_passed[1]
    content_list = human_msg.content

    image_parts = [p for p in content_list if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(image_parts) == 2, f"Expected 2 image parts, got {len(image_parts)}"

    # Verify both parts contain real base64 data URLs
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert len(image_parts[0]["image_url"]["url"]) > 100
    assert len(image_parts[1]["image_url"]["url"]) > 100

    # Verify audit recorded both images
    audit = result["audit"]
    assert audit.both_images_included is True
    assert len(audit.image_byte_lengths) == 2
    assert audit.api_call_status == "SUCCESS"


def test_candidate_coordinates_derived_from_geospatial_raster(tmp_path):
    """Test 3: Candidate coordinates and bounding boxes are derived directly from the georeferenced raster polygons."""
    service = ConstructionDiscoveryService(output_dir=tmp_path)
    bbox = [77.56, 28.125, 77.66, 28.215]

    aligned_data = service._read_and_align_rasters(
        bbox_wgs84=bbox,
        before_scene={"asset_urls": {"red": "mock://red.tif"}},
        after_scene={"asset_urls": {"red": "mock://red.tif"}},
    )

    detection = service._detect_changes(
        aligned_data=aligned_data,
        bbox_wgs84=bbox,
        min_change_area_m2=5000.0,
        run_dir=tmp_path,
    )

    candidates = detection["candidates"]
    assert len(candidates) >= 1

    for c in candidates:
        centroid = c["centroid"]
        lat = centroid["latitude"]
        lon = centroid["longitude"]

        # Coordinates must lie inside the requested bounding box
        assert 28.12 <= lat <= 28.22, f"Latitude {lat} outside Jewar bounds"
        assert 77.55 <= lon <= 77.67, f"Longitude {lon} outside Jewar bounds"
        assert "geometry" in c
        assert c["geometry"]["type"] in ("Polygon", "MultiPolygon")
        assert c["area_m2"] >= 5000.0


def test_failed_image_retrieval_or_vision_calls_produce_explicit_errors(tmp_path):
    """Test 4: Missing files or failed vision calls produce explicit VISION_NOT_EXECUTED, never silent successful fallback."""
    client = GeminiVerificationClient(api_key="mock_key")

    # Pass nonexistent image paths
    result = client.analyze_discovery_imagery(
        rgb_before_path=str(tmp_path / "nonexistent_before.png"),
        rgb_after_path=str(tmp_path / "nonexistent_after.png"),
        candidates=[],
        metrics={},
        scene_metadata={},
    )

    assert "VISION_NOT_EXECUTED" in result["summary"]
    assert result["audit"].api_call_status == "FAILED"
    assert result["audit"].parsed_response_status == "VISION_NOT_EXECUTED"
    assert result["audit"].vision_actually_executed is False

    # Test discovery service when EarthSearch finds no scenes
    service = ConstructionDiscoveryService(output_dir=tmp_path)
    with patch.object(service.earth_search, "search_scene", return_value=None):
        with pytest.raises(DiscoveryError) as exc_info:
            service.run_discovery(latitude=28.17, longitude=77.61)
        assert "No cloud-free Sentinel-2 scene" in str(exc_info.value)


def test_cloud_and_seasonal_artifacts_masked_by_scl(tmp_path):
    """Test 5: Cloud and cloud shadow pixels in SCL layer are masked out and excluded from construction candidate detection."""
    service = ConstructionDiscoveryService(output_dir=tmp_path)
    shape_grid = (100, 100)

    # Base arrays: NDVI would drop from 0.8 to -0.2 (delta = -1.0)
    r1 = np.full(shape_grid, 200.0, dtype=np.float32)
    n1 = np.full(shape_grid, 1800.0, dtype=np.float32)
    r2 = np.full(shape_grid, 1800.0, dtype=np.float32)
    n2 = np.full(shape_grid, 200.0, dtype=np.float32)

    # SCL marks entire image as cloud shadow (class 3) and cloud high probability (class 9)
    scl1 = np.full(shape_grid, 4, dtype=np.uint8)  # Vegetation
    scl2 = np.full(shape_grid, 9, dtype=np.uint8)  # Cloud high probability

    aligned_data = {
        "r1": r1, "g1": r1, "b1": r1, "n1": n1, "scl1": scl1,
        "r2": r2, "g2": r2, "b2": r2, "n2": n2, "scl2": scl2,
        "shape": shape_grid,
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_origin(77.56, 28.215, 0.001, 0.001),
    }

    detection = service._detect_changes(
        aligned_data=aligned_data,
        bbox_wgs84=[77.56, 28.125, 77.66, 28.215],
        min_change_area_m2=1000.0,
        run_dir=tmp_path,
    )

    # Because scl2 is cloud class 9, all pixels are masked as invalid; no fake construction candidates are returned
    assert detection["total_disturbed_area_m2"] == 0.0
    assert len(detection["candidates"]) == 0


def test_output_includes_provenance_and_geojson_polygons(tmp_path):
    """Test 6: Output report contains complete scene provenance and exportable GeoJSON FeatureCollection."""
    service = ConstructionDiscoveryService(output_dir=tmp_path)

    mock_b = {"scene_id": "S2A_TEST_BEFORE", "acquisition_date": "2021-03-20", "cloud_cover_percent": 1.5, "asset_urls": {"red": "mock://r.tif"}}
    mock_a = {"scene_id": "S2B_TEST_AFTER", "acquisition_date": "2025-03-24", "cloud_cover_percent": 0.2, "asset_urls": {"red": "mock://r.tif"}}

    with patch.object(service.earth_search, "search_scene", side_effect=[mock_b, mock_a]):
        report = service.run_discovery(latitude=28.17, longitude=77.61, radius_km=5.0)

    assert report.before_scene_id == "S2A_TEST_BEFORE"
    assert report.before_acquisition_date == "2021-03-20"
    assert report.after_scene_id == "S2B_TEST_AFTER"
    assert report.after_acquisition_date == "2025-03-24"
    assert report.resolution_m == 10.0

    fc = report.geojson_feature_collection
    assert fc.get("type") == "FeatureCollection"
    assert len(fc.get("features", [])) == report.candidates_count
    for feat in fc["features"]:
        assert feat["type"] == "Feature"
        assert "geometry" in feat
        assert "candidate_id" in feat["properties"]
        assert "area_m2" in feat["properties"]


def test_real_bsi_calculation_with_known_arrays(tmp_path):
    """Test 7: BSI and mean_delta_bsi are computed dynamically from actual SWIR/Red/NIR/Blue bands, not hardcoded."""
    service = ConstructionDiscoveryService(output_dir=tmp_path)

    # 1. Direct unit verification of calculate_bsi
    swir = np.array([[2000.0, 1000.0]], dtype=np.float32)
    red = np.array([[1000.0, 500.0]], dtype=np.float32)
    nir = np.array([[500.0, 2000.0]], dtype=np.float32)
    blue = np.array([[500.0, 500.0]], dtype=np.float32)

    bsi = service.calculate_bsi(swir, red, nir, blue)
    # Pixel 0: ((2000+1000) - (500+500)) / ((2000+1000) + (500+500)) = (3000 - 1000) / (3000 + 1000) = 2000 / 4000 = 0.50
    # Pixel 1: ((1000+500) - (2000+500)) / ((1000+500) + (2000+500)) = (1500 - 2500) / (1500 + 2500) = -1000 / 4000 = -0.25
    assert np.isclose(bsi[0, 0], 0.50, atol=1e-3)
    assert np.isclose(bsi[0, 1], -0.25, atol=1e-3)

    # 2. Integration verification in _detect_changes: delta_bsi calculation
    shape = (50, 50)
    # Before: dense vegetation / low bare soil
    # BSI1 should be negative
    sw1 = np.full(shape, 800.0, dtype=np.float32)
    r1 = np.full(shape, 400.0, dtype=np.float32)
    n1 = np.full(shape, 2400.0, dtype=np.float32)
    b1 = np.full(shape, 400.0, dtype=np.float32)
    # Before BSI = (1200 - 2800) / (1200 + 2800) = -1600 / 4000 = -0.40

    # After: bare cleared ground / construction
    # BSI2 should be positive
    sw2 = np.full(shape, 2400.0, dtype=np.float32)
    r2 = np.full(shape, 1200.0, dtype=np.float32)
    n2 = np.full(shape, 800.0, dtype=np.float32)
    b2 = np.full(shape, 600.0, dtype=np.float32)
    # After BSI = (3600 - 1400) / (3600 + 1400) = 2200 / 5000 = 0.44

    # Delta BSI expected = 0.44 - (-0.40) = 0.84 (definitely NOT hardcoded 0.18!)
    scl1 = np.full(shape, 4, dtype=np.uint8)
    scl2 = np.full(shape, 4, dtype=np.uint8)

    aligned_data = {
        "r1": r1, "g1": r1, "b1": b1, "n1": n1, "sw1": sw1, "scl1": scl1,
        "r2": r2, "g2": r2, "b2": b2, "n2": n2, "sw2": sw2, "scl2": scl2,
        "shape": shape,
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_origin(77.56, 28.215, 0.001, 0.001),
    }

    detection = service._detect_changes(
        aligned_data=aligned_data,
        bbox_wgs84=[77.56, 28.125, 77.66, 28.215],
        min_change_area_m2=100.0,
        run_dir=tmp_path,
    )

    candidates = detection["candidates"]
    assert len(candidates) >= 1
    cand = candidates[0]
    # Check that candidate delta BSI matches calculated value around 0.84, NOT 0.18
    assert abs(cand["mean_delta_bsi"] - 0.84) < 0.05
    assert cand["mean_delta_bsi"] != 0.18


def test_discovery_structured_output_valid_case(tmp_path):
    """Task 3: Test structured output with valid classifications (CONSTRUCTION and AGRICULTURAL_HARVEST)."""
    client = GeminiVerificationClient(api_key="mock_key")
    im1 = tmp_path / "b.png"
    im2 = tmp_path / "a.png"
    Image.new("RGB", (32, 32), "black").save(im1)
    Image.new("RGB", (32, 32), "white").save(im2)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = DiscoveryVisionStructuredOutput(
        overall_summary="Clear infrastructure earthworks alongside seasonal crop harvesting.",
        candidate_findings=[
            DiscoveryCandidateFinding(
                candidate_id="CANDIDATE-01",
                classification=CandidateClassification.CONSTRUCTION,
                confidence=0.92,
                visual_evidence="Linear paving and foundation footing excavations visible.",
                limitations="Sub-10m trench details unresolved.",
            ),
            DiscoveryCandidateFinding(
                candidate_id="CANDIDATE-02",
                classification=CandidateClassification.AGRICULTURAL_HARVEST,
                confidence=0.85,
                visual_evidence="Rectangular agricultural plot clearing with regular boundaries.",
                limitations="Could be initial perimeter grading.",
            ),
        ],
    )

    with patch.object(client, "_get_llm", return_value=mock_llm):
        res = client.analyze_discovery_imagery(
            rgb_before_path=str(im1),
            rgb_after_path=str(im2),
            candidates=[
                {"candidate_id": "CANDIDATE-01", "centroid": {"latitude": 28.17, "longitude": 77.61}, "area_hectares": 10.0, "area_m2": 100000.0},
                {"candidate_id": "CANDIDATE-02", "centroid": {"latitude": 28.18, "longitude": 77.62}, "area_hectares": 5.0, "area_m2": 50000.0},
            ],
            metrics={"total_disturbed_area_m2": 150000.0, "delta_ndvi_mean": -0.2},
            scene_metadata={"before_scene_id": "S2_B", "after_scene_id": "S2_A", "area_name": "Test Area"},
        )

    findings = res["findings"]
    assert len(findings) == 2
    f1 = next(f for f in findings if f.candidate_id == "CANDIDATE-01")
    f2 = next(f for f in findings if f.candidate_id == "CANDIDATE-02")
    assert f1.classification == CandidateClassification.CONSTRUCTION
    assert f1.confidence == 0.92
    assert "Linear paving" in f1.visual_evidence
    assert f2.classification == CandidateClassification.AGRICULTURAL_HARVEST
    assert f2.confidence == 0.85
    assert "Rectangular agricultural" in f2.visual_evidence
    assert res["audit"].vision_actually_executed is True


def test_discovery_structured_output_hallucinated_candidate_id_discarded(tmp_path):
    """Task 3: Test that hallucinated or non-existent candidate IDs returned by Gemini are discarded."""
    client = GeminiVerificationClient(api_key="mock_key")
    im1 = tmp_path / "b.png"
    im2 = tmp_path / "a.png"
    Image.new("RGB", (32, 32), "black").save(im1)
    Image.new("RGB", (32, 32), "white").save(im2)

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = DiscoveryVisionStructuredOutput(
        overall_summary="Summary with a hallucinated candidate ID.",
        candidate_findings=[
            DiscoveryCandidateFinding(
                candidate_id="CANDIDATE-01",
                classification=CandidateClassification.CONSTRUCTION,
                confidence=0.90,
                visual_evidence="Valid road grading.",
                limitations="None",
            ),
            DiscoveryCandidateFinding(
                candidate_id="CANDIDATE-FAKE-999",  # Hallucinated ID
                classification=CandidateClassification.WATER_BODY,
                confidence=0.99,
                visual_evidence="Hallucinated lake.",
                limitations="None",
            ),
        ],
    )

    with patch.object(client, "_get_llm", return_value=mock_llm):
        res = client.analyze_discovery_imagery(
            rgb_before_path=str(im1),
            rgb_after_path=str(im2),
            candidates=[
                {"candidate_id": "CANDIDATE-01", "centroid": {"latitude": 28.17, "longitude": 77.61}, "area_hectares": 10.0, "area_m2": 100000.0},
            ],
            metrics={"total_disturbed_area_m2": 100000.0, "delta_ndvi_mean": -0.2},
            scene_metadata={"before_scene_id": "S2_B", "after_scene_id": "S2_A", "area_name": "Test Area"},
        )

    findings = res["findings"]
    candidate_ids = [f.candidate_id for f in findings]
    assert "CANDIDATE-01" in candidate_ids
    assert "CANDIDATE-FAKE-999" not in candidate_ids
    assert len(findings) == 1


def test_discovery_structured_output_missing_candidate_marked_unclear(tmp_path):
    """Task 3: Test that candidates not returned by model are marked UNCLEAR with confidence 0.5."""
    client = GeminiVerificationClient(api_key="mock_key")
    im1 = tmp_path / "b.png"
    im2 = tmp_path / "a.png"
    Image.new("RGB", (32, 32), "black").save(im1)
    Image.new("RGB", (32, 32), "white").save(im2)

    mock_llm = MagicMock()
    # Model only returns CANDIDATE-01, leaving out CANDIDATE-02
    mock_llm.invoke.return_value = DiscoveryVisionStructuredOutput(
        overall_summary="Model missed the second candidate.",
        candidate_findings=[
            DiscoveryCandidateFinding(
                candidate_id="CANDIDATE-01",
                classification=CandidateClassification.CONSTRUCTION,
                confidence=0.88,
                visual_evidence="Runway foundation.",
                limitations="Low contrast.",
            ),
        ],
    )

    with patch.object(client, "_get_llm", return_value=mock_llm):
        res = client.analyze_discovery_imagery(
            rgb_before_path=str(im1),
            rgb_after_path=str(im2),
            candidates=[
                {"candidate_id": "CANDIDATE-01", "centroid": {"latitude": 28.17, "longitude": 77.61}, "area_hectares": 10.0, "area_m2": 100000.0},
                {"candidate_id": "CANDIDATE-02", "centroid": {"latitude": 28.18, "longitude": 77.62}, "area_hectares": 4.0, "area_m2": 40000.0},
            ],
            metrics={"total_disturbed_area_m2": 140000.0, "delta_ndvi_mean": -0.2},
            scene_metadata={"before_scene_id": "S2_B", "after_scene_id": "S2_A", "area_name": "Test Area"},
        )

    findings = res["findings"]
    assert len(findings) == 2
    f2 = next(f for f in findings if f.candidate_id == "CANDIDATE-02")
    assert f2.classification == CandidateClassification.UNCLEAR
    assert f2.confidence == 0.5


def test_discovery_structured_output_api_exception(tmp_path):
    """Task 3: Test that API exceptions return VISION_NOT_EXECUTED and vision_actually_executed=False."""
    client = GeminiVerificationClient(api_key="mock_key")
    im1 = tmp_path / "b.png"
    im2 = tmp_path / "a.png"
    Image.new("RGB", (32, 32), "black").save(im1)
    Image.new("RGB", (32, 32), "white").save(im2)

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Quota exhausted")

    with patch.object(client, "_get_llm", return_value=mock_llm):
        res = client.analyze_discovery_imagery(
            rgb_before_path=str(im1),
            rgb_after_path=str(im2),
            candidates=[
                {"candidate_id": "CANDIDATE-01", "centroid": {"latitude": 28.17, "longitude": 77.61}, "area_hectares": 10.0, "area_m2": 100000.0},
            ],
            metrics={"total_disturbed_area_m2": 100000.0, "delta_ndvi_mean": -0.2},
            scene_metadata={"before_scene_id": "S2_B", "after_scene_id": "S2_A", "area_name": "Test Area"},
        )

    assert "VISION_NOT_EXECUTED" in res["summary"]
    assert res["audit"].vision_actually_executed is False
    assert res["audit"].parsed_response_status == "VISION_NOT_EXECUTED"
    assert res["findings"] == []


