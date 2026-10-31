"""End-to-end case workflow with mocked OCR, Gemini, STAC and rasters. No network is used."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from schema_fixtures import png_bytes, scene_fixture, vision_response

from agents.construction_verification import case_workflow as cw
from agents.construction_verification.schemas import (
    ConstructionClaim,
    ExtractedProjectInfo,
    LocationInfo,
)
from agents.construction_verification.services.earth_search import (
    EarthSearchClient,
    EarthSearchError,
)
from agents.construction_verification.services.gemini_client import GeminiVerificationClient
from agents.construction_verification.services.ocr_client import OCRClient, OCRError
from agents.construction_verification.services.raster_processing import (
    AnalysisGrid,
    RasterProcessingError,
    RasterProcessor,
)

DOC_TEXT = "Project: Ring Road. Contractor: Acme Ltd. Claim: road surfacing completed."
CLAIM = "Road surfacing"
REVIEW_FORM = {
    "project_name": "Ring Road",
    "contractor": "Acme Ltd",
    "construction_type": "Road",
    "payment_amount": "INR 1,000",
    "latitude": "13.05",
    "longitude": "80.25",
    "start_date": "2024-01-10",
    "completion_date": "2024-06-20",
    "claims": CLAIM,
}
EXTRACTED = ExtractedProjectInfo(
    project_name="Ring Road",
    contractor="Acme Ltd",
    construction_type="Road",
    payment_amount="INR 1,000",
    location=LocationInfo(latitude=13.05, longitude=80.25),
    start_date="2024-01-10",
    completion_date="2024-06-20",
    claims=[ConstructionClaim(activity=CLAIM)],
)


@pytest.fixture
def production(tmp_path):
    """Production code paths with an isolated output directory; every service is mocked per test."""
    with (
        patch("core.settings.settings.USE_FAKE_MODEL", False),
        patch("core.settings.settings.CONSTRUCTION_OUTPUT_DIR", str(tmp_path)),
    ):
        yield tmp_path


def _case_with_document(content: bytes | None = None) -> dict:
    case = cw.new_case()
    cw.attach_document(case, "claim.png", content or png_bytes())
    return case


def _reviewed_case(**overrides) -> dict:
    case = _case_with_document()
    case["extracted_text"] = DOC_TEXT
    assert cw.apply_officer_review(case, {**REVIEW_FORM, **overrides}) == []
    return case


def _with_scenes(case: dict) -> dict:
    case["before_scene"] = scene_fixture("S2_BEFORE", "2024-01-05")
    case["after_scene"] = scene_fixture("S2_AFTER", "2024-06-25")
    return case


def _bands(nir: float, red: float, scl: int = 4, with_swir: bool = True):
    def read(self, asset_url, bbox, target_shape=None, grid=None, resampling=None):
        band = asset_url.rsplit("/", 1)[-1].split(".")[0]
        value = {"nir": nir, "red": red, "green": 500.0, "blue": 400.0, "swir16": 900.0, "scl": scl}[band]
        return np.full(grid.shape, value, dtype=np.float32), grid.transform

    return read


def _read_by_scene(before: tuple[float, float], after: tuple[float, float]):
    def read(self, asset_url, bbox, target_shape=None, grid=None, resampling=None):
        values = before if "S2_BEFORE" in asset_url else after
        band = asset_url.rsplit("/", 1)[-1].split(".")[0]
        value = {"nir": values[0], "red": values[1], "green": 500.0, "blue": 400.0, "swir16": 900.0, "scl": 4}[band]
        return np.full(grid.shape, value, dtype=np.float32), grid.transform

    return read


def test_01_valid_document_upload_is_stored_without_network(production):
    case = cw.new_case()
    assert cw.attach_document(case, "claim.pdf", b"%PDF-1.7 content") is True
    stored = production / "uploads" / f"{case['case_id']}_{case['document_sha256'][:12]}.pdf"
    assert stored.read_bytes() == b"%PDF-1.7 content"
    assert case["document_paths"] == [str(stored)]
    assert case["document_name"] == "claim.pdf"


def test_01b_unsupported_upload_is_rejected(production):
    with pytest.raises(ValueError, match="Unsupported"):
        cw.attach_document(cw.new_case(), "claim.exe", b"x")


def test_01b2_corrupt_image_is_rejected_at_upload(production):
    case = cw.new_case()
    with pytest.raises(ValueError, match="not a readable image"):
        cw.attach_document(case, "claim.png", b"not an image")
    assert case["document_paths"] == []


def test_01c_rerun_with_same_file_does_not_reset_case(production):
    case = _case_with_document(png_bytes(10))
    case["extracted_text"] = "already extracted"
    assert cw.attach_document(case, "claim.png", png_bytes(10)) is False
    assert case["extracted_text"] == "already extracted"


def test_01d_replacing_document_clears_all_derived_results(production):
    case = _with_scenes(_reviewed_case())
    case["report"] = {"case_id": case["case_id"]}
    assert cw.attach_document(case, "other.png", png_bytes(200)) is True
    for key in ("extracted_text", "project_name", "bounding_box", "before_scene", "report"):
        assert key not in case


def test_02_ocr_response_reaches_extraction(production):
    case = _case_with_document()
    with patch.object(OCRClient, "_call_ocr_space", AsyncMock(return_value=DOC_TEXT)) as ocr:
        assert cw.run_stage(case, "ocr") == []
    ocr.assert_awaited_once()
    assert case["extracted_text"] == DOC_TEXT

    with patch.object(
        GeminiVerificationClient, "extract_project_info", AsyncMock(return_value=EXTRACTED)
    ) as extract:
        assert cw.run_stage(case, "extract") == []
    extract.assert_awaited_once_with(DOC_TEXT)


def test_02b_ocr_failure_never_inserts_example_text(production):
    case = _case_with_document()
    with patch.object(OCRClient, "_call_ocr_space", AsyncMock(side_effect=OCRError("HTTP 500"))):
        errors = cw.run_stage(case, "ocr")
    assert any("HTTP 500" in e for e in errors)
    assert "extracted_text" not in case
    assert "HTTP 500" in case["ocr_error"]


def test_02c_missing_ocr_key_fails_clearly_for_scanned_document(production):
    path = production / "scan.png"
    path.write_bytes(b"\x89PNG")
    with pytest.raises(OCRError, match="OCR_API_KEY"):
        asyncio.run(_no_key_ocr(path))


async def _no_key_ocr(path):
    client = OCRClient()
    client.api_key = None
    return await client.extract_text_from_file(path)


def test_03_structured_extraction_is_parsed_into_case_and_form(production):
    case = _case_with_document()
    case["extracted_text"] = DOC_TEXT
    client_patch = patch.object(
        GeminiVerificationClient,
        "_ainvoke_with_fallback",
        AsyncMock(return_value=(EXTRACTED.model_dump(), "model")),
    )
    with client_patch as invoke:
        assert cw.run_stage(case, "extract") == []
    assert invoke.await_args.kwargs["structured_schema"] is ExtractedProjectInfo
    assert DOC_TEXT in invoke.await_args.args[0][1].content

    values = cw.review_form_values(case)
    assert values["project_name"] == "Ring Road"
    assert values["contractor"] == "Acme Ltd"
    assert values["payment_amount"] == "INR 1,000"
    assert values["latitude"] == "13.05"
    assert values["claims"] == CLAIM
    assert case["form_revision"] == 1


def test_03b_gemini_extraction_failure_does_not_invent_details(production):
    case = _case_with_document()
    case["extracted_text"] = "Runway works near the airport"
    with patch.object(
        GeminiVerificationClient, "_ainvoke_with_fallback", AsyncMock(side_effect=RuntimeError("503"))
    ):
        errors = cw.run_stage(case, "extract")
    assert any("503" in e for e in errors)
    assert "503" in case["extraction_error"]
    assert not case.get("construction_claims")
    assert not case.get("project_name")


def test_04_missing_coordinates_are_detected_not_invented(production):
    case = _case_with_document()
    case["extracted_text"] = DOC_TEXT
    no_location = EXTRACTED.model_copy(update={"location": None})
    with patch.object(
        GeminiVerificationClient, "_ainvoke_with_fallback", AsyncMock(return_value=(no_location, "m"))
    ):
        errors = cw.run_stage(case, "extract")
    assert any("site_coordinates" in e for e in errors)
    assert case["construction_location"]["latitude"] is None
    assert not case.get("bounding_box")
    missing = cw.missing_review_fields(case)
    assert "Latitude" in missing and "Longitude" in missing

    review = cw.apply_officer_review(case, {**REVIEW_FORM, "latitude": "", "longitude": ""})
    assert "Latitude is required." in review
    assert not case.get("review_confirmed")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("latitude", "91", "Latitude must be between"),
        ("longitude", "abc", "Longitude must be a decimal"),
        ("start_date", "10/01/2024", "valid date"),
        ("start_date", "2025-01-01", "must not be after"),
        ("claims", "  \n ", "At least one"),
    ],
)
def test_04b_invalid_review_input_is_rejected(production, field, value, message):
    case = _case_with_document()
    errors = cw.apply_officer_review(case, {**REVIEW_FORM, field: value})
    assert any(message in e for e in errors)
    assert not case.get("review_confirmed")
    assert "bounding_box" not in case


def test_05_coordinate_correction_reaches_stac_and_invalidates(production):
    case = _with_scenes(_reviewed_case())
    case["satellite_metrics"] = {"delta_ndvi": -0.5}
    case["report"] = {"case_id": case["case_id"]}

    assert cw.apply_officer_review(case, {**REVIEW_FORM, "latitude": "22.5", "longitude": "88.3"}) == []
    assert "before_scene" not in case and "satellite_metrics" not in case and "report" not in case
    assert case["construction_location"]["latitude"] == 22.5

    with patch.object(EarthSearchClient, "find_best_scene", return_value=scene_fixture("S", "2024-01-01")) as find:
        assert cw.run_stage(case, "retrieve") == []
    assert find.call_args_list[0].kwargs["bbox"] == [88.295, 22.495, 88.305, 22.505]


def test_06_date_correction_reaches_stac_and_invalidates(production):
    case = _with_scenes(_reviewed_case())
    assert cw.apply_officer_review(case, {**REVIEW_FORM, "start_date": "2023/11/05"}) == []
    assert case["start_date"] == "2023-11-05"
    assert "before_scene" not in case

    with patch.object(EarthSearchClient, "find_best_scene", return_value=scene_fixture("S", "2024-01-01")) as find:
        cw.run_stage(case, "retrieve")
    assert find.call_args_list[0].kwargs["target_date"] == "2023-11-05"
    assert find.call_args_list[1].kwargs["target_date"] == "2024-06-20"


def test_06b_claim_edit_keeps_measurements_but_drops_vision(production):
    case = _with_scenes(_reviewed_case())
    case.update(satellite_metrics={"delta_ndvi": -0.5}, vision_audit={"x": 1}, report={"a": 1})
    assert cw.apply_officer_review(case, {**REVIEW_FORM, "claims": "New claim"}) == []
    assert case["satellite_metrics"] == {"delta_ndvi": -0.5}
    assert "before_scene" in case
    assert "vision_audit" not in case and "report" not in case


def test_06c_retrieval_is_gated_on_officer_confirmation(production):
    case = _case_with_document()
    with patch.object(EarthSearchClient, "find_best_scene") as find:
        errors = cw.run_stage(case, "retrieve")
    find.assert_not_called()
    assert "confirm" in errors[0]


def test_07_stac_scene_metadata_reaches_raster_pipeline(production):
    case = _reviewed_case()
    before, after = scene_fixture("S2_BEFORE", "2024-01-05"), scene_fixture("S2_AFTER", "2024-06-25")
    with patch.object(EarthSearchClient, "find_best_scene", side_effect=[before, after]):
        cw.run_stage(case, "retrieve")

    urls = []

    def read(self, asset_url, bbox, target_shape=None, grid=None, resampling=None):
        urls.append(asset_url)
        return np.full(grid.shape, 2 if "scl" in asset_url else 1000, dtype=np.float32), None

    with (
        patch.object(RasterProcessor, "read_window_band", read),
        patch.object(GeminiVerificationClient, "_ainvoke_with_fallback", AsyncMock(side_effect=RuntimeError("x"))),
    ):
        cw.run_stage(case, "analyze")
    assert set(before["asset_urls"].values()) | set(after["asset_urls"].values()) == set(urls)
    assert case["satellite_metrics"]["before_scene_id"] == "S2_BEFORE"
    assert case["satellite_metrics"]["after_scene_id"] == "S2_AFTER"
    assert case["satellite_metrics"]["synthetic"] is False


def test_08_synthetic_arrays_produce_correct_ndvi(production):
    processor = RasterProcessor(output_dir=production)
    before = scene_fixture("S2_BEFORE", "2024-01-05")
    after = scene_fixture("S2_AFTER", "2024-06-25")
    with patch.object(RasterProcessor, "read_window_band", _read_by_scene((3000.0, 1000.0), (1500.0, 1500.0))):
        metrics = processor.analyze_change("C1", [80.245, 13.045, 80.255, 13.055], before, after)
    assert metrics["mean_ndvi_before"] == pytest.approx(0.5)
    assert metrics["mean_ndvi_after"] == pytest.approx(0.0)
    assert metrics["delta_ndvi"] == pytest.approx(-0.5)
    assert metrics["vegetation_loss_fraction"] == 1.0
    assert metrics["valid_observation_coverage_pct"] == 100.0
    height, width = metrics["raster_shape"]
    assert metrics["potential_disturbed_area_m2"] == height * width * 100.0
    # ±0.005° is about 1.1 km, so 10 m pixels must give roughly 108 pixels per side.
    assert 100 < height < 120 and 100 < width < 120
    assert metrics["delta_bsi"] is not None


def test_08a_declared_offset_metadata_is_not_subtracted_from_pixels(production):
    processor = RasterProcessor(output_dir=production)
    before = scene_fixture("S2_BEFORE", "2024-01-05")
    after = scene_fixture("S2_AFTER", "2024-06-25")
    before["properties"]["raster_offset_declared"] = -0.1
    with patch.object(RasterProcessor, "read_window_band", _read_by_scene((2371.0, 851.0), (2787.0, 1550.0))):
        metrics = processor.analyze_change("C1", [80.245, 13.045, 80.255, 13.055], before, after)
    # Median values measured from live scenes, which give plausible NDVI only without an offset.
    assert metrics["mean_ndvi_before"] == pytest.approx(0.4717, abs=0.001)
    assert metrics["mean_ndvi_after"] == pytest.approx(0.2852, abs=0.001)


def test_08b_missing_swir_reports_bsi_as_absent_not_zero(production):
    processor = RasterProcessor(output_dir=production)
    before = scene_fixture("S2_BEFORE", "2024-01-05")
    after = scene_fixture("S2_AFTER", "2024-06-25")
    del before["asset_urls"]["swir16"]
    with patch.object(RasterProcessor, "read_window_band", _read_by_scene((3000.0, 1000.0), (1500.0, 1500.0))):
        metrics = processor.analyze_change("C1", [80.245, 13.045, 80.255, 13.055], before, after)
    assert metrics["delta_bsi"] is None and metrics["mean_bsi_before"] is None


def test_08c_fully_cloudy_site_fails_instead_of_reporting_numbers(production):
    processor = RasterProcessor(output_dir=production)
    with patch.object(RasterProcessor, "read_window_band", _bands(3000.0, 1000.0, scl=9)):
        with pytest.raises(RasterProcessingError, match="obscured"):
            processor.analyze_change(
                "C1",
                [80.245, 13.045, 80.255, 13.055],
                scene_fixture("S2_BEFORE", "2024-01-05"),
                scene_fixture("S2_AFTER", "2024-06-25"),
            )


def _write_tif(path, crs, value, bbox, resolution=10.0):
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox)
    width = int((right - left) / resolution) + 1
    height = int((top - bottom) / resolution) + 1
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32", crs=crs,
        transform=from_origin(left, top, resolution, resolution),
    ) as dst:
        data = np.full((height, width), value, dtype=np.float32)
        # Marker in the western half so a misaligned read would change the mean.
        data[:, : width // 2] = value / 2
        dst.write(data, 1)


def test_08d_rasters_in_different_utm_zones_align_on_one_grid(production, tmp_path):
    bbox = [78.0 - 0.004, 13.0, 78.0 + 0.004, 13.008]  # straddles the UTM 43/44 boundary
    wider = [bbox[0] - 0.01, bbox[1] - 0.01, bbox[2] + 0.01, bbox[3] + 0.01]
    zone43 = tmp_path / "z43.tif"
    zone44 = tmp_path / "z44.tif"
    _write_tif(zone43, "EPSG:32643", 1000.0, wider)
    _write_tif(zone44, "EPSG:32644", 1000.0, wider)

    processor = RasterProcessor(output_dir=tmp_path)
    grid = AnalysisGrid.from_bbox(bbox)
    arr43, _ = processor.read_window_band(str(zone43), bbox, grid=grid)
    arr44, _ = processor.read_window_band(str(zone44), bbox, grid=grid)

    assert arr43.shape == arr44.shape == grid.shape
    assert np.isclose(arr43, arr44, atol=1.0).mean() > 0.95


def test_09_both_image_payloads_are_sent_to_gemini_vision(production):
    case = _with_scenes(_reviewed_case())
    invoke = AsyncMock(return_value=(vision_response(CLAIM, "SUPPORTED"), "gemini-test"))
    with (
        patch.object(RasterProcessor, "read_window_band", _read_by_scene((3000.0, 1000.0), (1500.0, 1500.0))),
        patch.object(GeminiVerificationClient, "_ainvoke_with_fallback", invoke),
    ):
        assert cw.run_stage(case, "analyze") == []

    messages = invoke.await_args.args[0]
    content = messages[1].content
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == 2
    metrics = case["satellite_metrics"]
    for part, key in zip(images, ("rgb_before_preview_path", "rgb_after_preview_path")):
        header, encoded = part["image_url"]["url"].split(",", 1)
        assert header == "data:image/png;base64"
        with open(metrics[key], "rb") as f:
            assert base64.b64decode(encoded) == f.read()
    assert images[0] != images[1]
    assert CLAIM in content[0]["text"]
    assert str(metrics["delta_ndvi"]) in content[0]["text"]
    json.dumps(content)

    audit = case["vision_audit"]
    assert audit["both_images_included"] is True
    assert len(audit["image_byte_lengths"]) == 2
    assert audit["before_scene_id"] == "S2_BEFORE"
    assert audit["vision_actually_executed"] is True


def test_09b_vision_schema_sent_to_model_has_no_audit_or_union_fields(production):
    schema_text = json.dumps(
        __import__(
            "agents.construction_verification.schemas", fromlist=["VisionModelResponse"]
        ).VisionModelResponse.model_json_schema()
    )
    assert "vision_actually_executed" not in schema_text
    assert "prefixItems" not in schema_text
    assert "supports_claim" not in schema_text


@pytest.mark.parametrize(
    "side_effect",
    [RuntimeError("500 backend"), [(None, "m")], [({"findings": []}, "m")], [("not structured", "m")]],
)
def test_10_gemini_failure_produces_vision_not_executed(production, side_effect):
    case = _with_scenes(_reviewed_case())
    with (
        patch.object(RasterProcessor, "read_window_band", _read_by_scene((3000.0, 1000.0), (1500.0, 1500.0))),
        patch.object(GeminiVerificationClient, "_ainvoke_with_fallback", AsyncMock(side_effect=side_effect)),
    ):
        cw.run_stage(case, "analyze")
    assert case["vision_audit"]["vision_actually_executed"] is False
    assert [r["status"] for r in case["verification_results"]] == ["VISION_NOT_EXECUTED"]
    assert case["overall_status"] == "VISION_NOT_EXECUTED_REQUIRES_INSPECTION"
    assert case["verification_results"][0]["observations"] == []
    # Deterministic measurements survive so the officer still sees real evidence.
    assert case["report"]["satellite_metrics"]["delta_ndvi"] == pytest.approx(-0.5)
    assert any("did not assess" in lim for lim in case["report"]["evidence_limitations"])


def test_10b_missing_gemini_key_produces_vision_not_executed(production):
    case = _with_scenes(_reviewed_case())
    with (
        patch.object(RasterProcessor, "read_window_band", _read_by_scene((3000.0, 1000.0), (1500.0, 1500.0))),
        patch("core.settings.settings.GEMINI_API_KEY", None),
        patch("core.settings.settings.GOOGLE_API_KEY", None),
    ):
        cw.run_stage(case, "analyze")
    assert [r["status"] for r in case["verification_results"]] == ["VISION_NOT_EXECUTED"]
    assert "GEMINI_API_KEY" in case["vision_audit"]["error_message"]


def test_10c_vision_retry_does_not_recompute_raster_and_recovers(production):
    case = _with_scenes(_reviewed_case())
    read = _read_by_scene((3000.0, 1000.0), (1500.0, 1500.0))
    with (
        patch.object(RasterProcessor, "read_window_band", read),
        patch.object(GeminiVerificationClient, "_ainvoke_with_fallback", AsyncMock(side_effect=RuntimeError("429"))),
    ):
        cw.run_stage(case, "analyze")
    assert not cw.stage_complete(case, "analyze")

    with (
        patch.object(RasterProcessor, "analyze_change") as analyze,
        patch.object(
            GeminiVerificationClient,
            "_ainvoke_with_fallback",
            AsyncMock(return_value=(vision_response(CLAIM, "NOT_SUPPORTED"), "m")),
        ),
    ):
        cw.run_stage(case, "analyze")
    analyze.assert_not_called()
    assert case["verification_results"][0]["status"] == "NOT_SUPPORTED"


def test_11_retrieval_failure_generates_no_synthetic_evidence(production):
    case = _reviewed_case()
    with patch.object(EarthSearchClient, "find_best_scene", side_effect=EarthSearchError("STAC down")):
        errors = cw.run_stage(case, "retrieve")
    assert any("STAC down" in e for e in errors)
    assert case["before_scene"] == {} and case["after_scene"] == {}
    assert not cw.stage_complete(case, "retrieve")

    with patch.object(RasterProcessor, "analyze_change") as analyze:
        errors = cw.run_stage(case, "analyze")
    analyze.assert_not_called()
    assert "required before analysis" in errors[0]
    assert "satellite_metrics" not in case and "report" not in case


def test_11b_missing_scene_is_reported(production):
    case = _reviewed_case()
    with patch.object(EarthSearchClient, "find_best_scene", side_effect=[scene_fixture("S", "2024-01-01"), None]):
        errors = cw.run_stage(case, "retrieve")
    assert any("after_scene: False" in e for e in errors)
    assert not cw.stage_complete(case, "retrieve")


def test_11c_raster_failure_reaches_report_without_measurements(production):
    case = _with_scenes(_reviewed_case())
    with patch.object(RasterProcessor, "read_window_band", side_effect=RasterProcessingError("S3 403")):
        errors = cw.run_stage(case, "analyze")
    assert any("S3 403" in e for e in errors)
    assert case["report"]["satellite_metrics"] is None
    assert case["verification_results"][0]["status"] == "VISION_NOT_EXECUTED"
    assert any("S3 403" in lim for lim in case["report"]["evidence_limitations"])


def test_11d_mock_scene_is_refused_in_production(production):
    case = _reviewed_case()
    fake = {
        "scene_id": "S2A_SYNTHETIC",
        "acquisition_date": "2024-01-01",
        "cloud_cover_percent": 1.0,
        "asset_urls": {"red": "mock://red.tif"},
        "properties": {"synthetic": True},
    }
    case.update(before_scene=fake, after_scene=fake)
    errors = cw.run_stage(case, "analyze")
    assert any("synthetic" in e.lower() for e in errors)
    assert case["report"]["satellite_metrics"] is None
    assert case["report"]["before_scene"] is None
    assert case["report"]["evidence_mode"] == "LIVE_SERVICES"
    assert case["verification_results"][0]["status"] == "VISION_NOT_EXECUTED"


def test_11e_test_mode_report_is_labelled_synthetic(tmp_path):
    with (
        patch("core.settings.settings.USE_FAKE_MODEL", True),
        patch("core.settings.settings.CONSTRUCTION_OUTPUT_DIR", str(tmp_path)),
    ):
        case = _reviewed_case()
        assert cw.run_stage(case, "retrieve") == []
        assert case["before_scene"]["properties"]["synthetic"] is True
        cw.run_stage(case, "analyze")
    assert case["satellite_metrics"]["synthetic"] is True
    assert case["report"]["evidence_mode"] == "SYNTHETIC_TEST_DATA"
    assert "SYNTHETIC" in case["report"]["evidence_limitations"][0]
    assert case["vision_audit"]["vision_actually_executed"] is False
    assert {r["status"] for r in case["verification_results"]} == {"VISION_NOT_EXECUTED"}


def test_12_report_reflects_actual_findings(production):
    case = _with_scenes(_reviewed_case(claims=f"{CLAIM}\nDeep piling\nUnreturned claim"))
    response = {
        "findings": [
            {"claim": CLAIM, "status": "SUPPORTED", "visual_observations": ["New linear surface"],
             "reasoning": "Corridor visible", "limitations": ["10 m"], "recommended_follow_up": "Spot check"},
            {"claim": "Deep piling", "status": "INSUFFICIENT_EVIDENCE", "visual_observations": [],
             "reasoning": "Sub-surface", "limitations": ["Not visible"], "recommended_follow_up": "Inspect"},
            {"claim": "", "status": "SUPPORTED", "visual_observations": [],
             "reasoning": "blank claim must match nothing", "limitations": [], "recommended_follow_up": "x"},
        ],
        "general_observations": "Corridor appeared",
    }
    with (
        patch.object(RasterProcessor, "read_window_band", _read_by_scene((3000.0, 1000.0), (1500.0, 1500.0))),
        patch.object(GeminiVerificationClient, "_ainvoke_with_fallback", AsyncMock(return_value=(response, "m"))),
    ):
        cw.run_stage(case, "analyze")

    report = case["report"]
    by_claim = {c["claim"]: c for c in report["claims_summary"]}
    assert by_claim[CLAIM]["status"] == "SUPPORTED"
    assert by_claim[CLAIM]["observations"] == ["New linear surface"]
    assert by_claim[CLAIM]["reason"] == "Corridor visible"
    assert by_claim[CLAIM]["limitations"] == ["10 m"]
    assert by_claim[CLAIM]["recommended_next_action"] == "Spot check"
    assert by_claim["Deep piling"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert by_claim["Unreturned claim"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["overall_status"] == "INSUFFICIENT_EVIDENCE_REQUIRES_INSPECTION"
    assert report["contractor"] == "Acme Ltd"
    assert report["payment_amount"] == "INR 1,000"
    assert report["before_scene"]["scene_id"] == "S2_BEFORE"
    assert report["satellite_metrics"]["delta_ndvi"] == pytest.approx(-0.5)
    assert report["general_observations"] == "Corridor appeared"
    assert report["officer_decision"] is None

    saved = json.loads((production / "reports" / f"{case['case_id']}_report.json").read_text())
    assert saved["overall_status"] == report["overall_status"]


def test_12b_no_claims_never_reports_supported(production):
    from agents.construction_verification.nodes.verification import verification_node

    result = asyncio.run(verification_node({"case_id": "C", "construction_claims": []}, {}))
    assert result["overall_status"] == "NO_CLAIMS_ASSESSED"


def test_12c_all_supported_still_does_not_authorize_payment(production):
    case = _with_scenes(_reviewed_case())
    with (
        patch.object(RasterProcessor, "read_window_band", _read_by_scene((3000.0, 1000.0), (1500.0, 1500.0))),
        patch.object(
            GeminiVerificationClient,
            "_ainvoke_with_fallback",
            AsyncMock(return_value=(vision_response(CLAIM, "SUPPORTED"), "m")),
        ),
    ):
        cw.run_stage(case, "analyze")
    assert case["overall_status"] == "SATELLITE_EVIDENCE_SUPPORTED"
    assert not case.get("officer_decision")
    assert case["report"]["officer_decision"] is None
    assert not (production / "decisions").exists()


def _finished_case(production) -> dict:
    case = _with_scenes(_reviewed_case())
    with (
        patch.object(RasterProcessor, "read_window_band", _read_by_scene((3000.0, 1000.0), (1500.0, 1500.0))),
        patch.object(
            GeminiVerificationClient,
            "_ainvoke_with_fallback",
            AsyncMock(return_value=(vision_response(CLAIM, "SUPPORTED"), "m")),
        ),
    ):
        cw.run_stage(case, "analyze")
    return case


def test_13_officer_decision_is_recorded_separately(production):
    case = _finished_case(production)
    report_before = json.dumps(case["report"], sort_keys=True, default=str)

    record = cw.record_decision(case, "REQUEST_INSPECTION", " OFF-7 ", " Needs site visit ")

    path = production / "decisions" / f"{case['case_id']}_decision.json"
    saved = json.loads(path.read_text())
    assert saved == record
    assert saved["decision"] == "REQUEST_INSPECTION"
    assert saved["officer_name_or_id"] == "OFF-7"
    assert saved["justification"] == "Needs site visit"
    assert len(saved["report_sha256"]) == 64
    assert json.dumps(case["report"], sort_keys=True, default=str) == report_before
    report_file = json.loads((production / "reports" / f"{case['case_id']}_report.json").read_text())
    assert report_file["officer_decision"] is None


@pytest.mark.parametrize(
    "decision,officer,justification,message",
    [
        ("APPROVE", "", "reason", "identifier"),
        ("APPROVE", "OFF", "   ", "Justification"),
        ("AUTO_PAY", "OFF", "reason", "AUTO_PAY"),
    ],
)
def test_13b_incomplete_decision_is_rejected(production, decision, officer, justification, message):
    case = _finished_case(production)
    with pytest.raises(ValueError, match=message):
        cw.record_decision(case, decision, officer, justification)
    assert not case.get("officer_decision")
    assert not (production / "decisions" / f"{case['case_id']}_decision.json").exists()


def test_13c_decision_requires_report_and_cannot_be_overwritten(production):
    with pytest.raises(cw.DecisionError, match="report is required"):
        cw.record_decision(cw.new_case(), "HOLD", "OFF", "reason")

    case = _finished_case(production)
    cw.record_decision(case, "HOLD", "OFF", "reason")
    with pytest.raises(cw.DecisionError, match="already"):
        cw.record_decision(case, "APPROVE", "OFF", "changed my mind")
    assert case["officer_decision"]["decision"] == "HOLD"

    errors = cw.apply_officer_review(case, {**REVIEW_FORM, "latitude": "1"})
    assert "recorded decision" in errors[0]
    assert case["construction_location"]["latitude"] == 13.05


def test_14_new_case_inherits_nothing(production):
    first = _finished_case(production)
    cw.record_decision(first, "HOLD", "OFF", "reason")

    second = cw.new_case()
    assert second["case_id"] != first["case_id"]
    assert set(second) == {"case_id", "document_paths", "form_revision", "stage_errors"}
    assert second["document_paths"] == [] and second["stage_errors"] == {}
    assert second["stage_errors"] is not first["stage_errors"]
    assert cw.review_form_values(second)["latitude"] == ""
    assert not cw.stage_complete(second, "report")


def test_15_completed_stages_are_not_repeated_on_rerun(production):
    case = _reviewed_case()
    scene = scene_fixture("S", "2024-01-01")
    with patch.object(EarthSearchClient, "find_best_scene", return_value=scene) as find:
        cw.run_stage(case, "retrieve")
        cw.run_stage(case, "retrieve")
        cw.run_stage(case, "retrieve")
    assert find.call_count == 2  # one before and one after search, once only

    case = _case_with_document()
    with patch.object(OCRClient, "_call_ocr_space", AsyncMock(return_value=DOC_TEXT)) as ocr:
        cw.run_stage(case, "ocr")
        cw.run_stage(case, "ocr")
    ocr.assert_awaited_once()

    finished = _finished_case(production)
    with patch.object(GeminiVerificationClient, "_ainvoke_with_fallback", AsyncMock()) as invoke:
        cw.run_stage(finished, "analyze")
    invoke.assert_not_called()


def test_15b_scanned_pdf_uses_ocr_and_text_pdf_stays_local(production, tmp_path):
    pypdf = pytest.importorskip("pypdf")
    blank = tmp_path / "scan.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with open(blank, "wb") as f:
        writer.write(f)
    client = OCRClient(api_key="placeholder")
    with patch.object(OCRClient, "_call_ocr_space", AsyncMock(return_value="ocr text")) as ocr:
        assert asyncio.run(client.extract_text_from_file(blank)) == "ocr text"
    ocr.assert_awaited_once()

    with (
        patch.object(OCRClient, "_extract_pdf_text", return_value="embedded text from a digital pdf"),
        patch.object(OCRClient, "_call_ocr_space", AsyncMock()) as ocr,
    ):
        assert "embedded" in asyncio.run(client.extract_text_from_file(blank))
    ocr.assert_not_called()


def test_15c_invalid_bbox_never_reaches_stac(production):
    case = _reviewed_case()
    case["bounding_box"] = [200.0, 13.0, 201.0, 14.0]
    with patch.object(EarthSearchClient, "find_best_scene") as find:
        errors = cw.run_stage(case, "retrieve")
    find.assert_not_called()
    assert any("Invalid coordinates" in e for e in errors)
