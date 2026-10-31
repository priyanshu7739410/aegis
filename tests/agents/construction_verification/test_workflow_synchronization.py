import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from agents.agents import get_agent
from agents.construction_verification.graph import construction_verification_agent
from agents.construction_verification.schemas import LocationInfo, ExtractedProjectInfo
from agents.construction_verification.services.ocr_client import OCRClient, OCRError
from service.service import app
from service.construction import load_case_record, save_case_record


ROAD_LETTER_TEXT = """
OFFICIAL REIMBURSEMENT REQUEST & COMPLETION LETTER
Government of West Bengal - Public Works Department (Rural Roads)
Project: Construction and Improvement of Rural Link Road, Barasat–Kazipara
Location: Latitude 22.7215, Longitude 88.4812
Construction Type: Bituminous road construction and drainage improvement
Road length: 1.8 km, Road width: 5.5 m
Start Date: 2025-02-01
Completion Date: 2026-09-30
Claim 1: Bituminous macadam road construction (1.8 km length, 5.5 m width) - 100% completed.
Claim 2: Reinforced side drain excavation and concrete culvert installation - 100% completed.
Contractor: Bengal Infrastructure Developers Ltd.
Final payment milestone verification requested.
"""


@pytest.mark.asyncio
async def test_regression_1_uploaded_document_reaches_ocr(tmp_path):
    """Test 1: Uploaded document reaches OCR client without data loss."""
    test_pdf = tmp_path / "barasat_road.pdf"
    test_pdf.write_text(ROAD_LETTER_TEXT, encoding="utf-8")

    client = OCRClient()
    # Mock OCR.space to ensure it is called with the actual uploaded file
    with patch.object(client, "_call_ocr_space", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ROAD_LETTER_TEXT
        text = await client.extract_text_from_file(test_pdf)

        assert "Barasat–Kazipara" in text
        assert "22.7215" in text
        mock_call.assert_awaited_once_with(test_pdf)


@pytest.mark.asyncio
async def test_regression_2_ocr_output_reaches_langgraph():
    """Test 2: OCR output reaches LangGraph and updates state with extracted project info."""
    inputs = {
        "case_id": "CASE-TEST-BARASAT-001",
        "extracted_text": ROAD_LETTER_TEXT,
    }
    with patch("agents.construction_verification.nodes.satellite_search.EarthSearchClient") as MockSTAC, \
         patch("agents.construction_verification.nodes.satellite_analysis.RasterProcessor") as MockProc:
        instance = MockSTAC.return_value
        instance.find_best_scene.side_effect = [
            {"scene_id": "S2A_BEFORE", "acquisition_date": "2025-01-20", "cloud_cover_percent": 2.5, "asset_urls": {"red": "mock_r", "nir": "mock_n", "visual": "mock_v"}},
            {"scene_id": "S2A_AFTER", "acquisition_date": "2026-09-15", "cloud_cover_percent": 4.1, "asset_urls": {"red": "mock_r2", "nir": "mock_n2", "visual": "mock_v2"}},
        ]
        proc_inst = MockProc.return_value
        proc_inst.analyze_change.return_value = {
            "mean_ndvi_before": 0.52, "mean_ndvi_after": 0.31, "delta_ndvi": -0.21,
            "vegetation_loss_fraction": 0.38, "potential_disturbed_area_m2": 11500.0,
            "mean_bsi_before": -0.03, "mean_bsi_after": 0.15, "delta_bsi": 0.18,
            "valid_observation_coverage_pct": 100.0,
            "rgb_before_preview_path": "mock_b.png", "rgb_after_preview_path": "mock_a.png",
        }
        result = await construction_verification_agent.ainvoke(
            inputs, config={"configurable": {"thread_id": inputs["case_id"]}}
        )

    assert "project_name" in result
    assert "Barasat–Kazipara" in result["project_name"]
    assert result["construction_location"]["latitude"] == 22.7215
    assert result["construction_location"]["longitude"] == 88.4812
    assert result["start_date"] == "2025-02-01"
    assert result["completion_date"] == "2026-09-30"
    assert len(result["construction_claims"]) >= 1


@pytest.mark.asyncio
async def test_regression_3_ocr_text_not_replaced_with_demo_data(tmp_path):
    """Test 3: Failed or image OCR does not substitute generic NH-44 sample data."""
    binary_img = tmp_path / "corrupted.png"
    binary_img.write_bytes(b"\x89PNG\r\n\x1a\nCorruptBinaryData")

    client = OCRClient(api_key=None)  # simulate offline / no key
    with pytest.raises(OCRError) as exc_info:
        await client.extract_text_from_file(binary_img)

    # Must raise clear error, NOT return NH-44
    err_text = str(exc_info.value)
    assert "NH-44" not in err_text
    assert "Bangalore" not in err_text


@pytest.mark.asyncio
async def test_regression_4_manual_coordinates_update_graph_state():
    """Test 4: Manual coordinate inputs update graph state and location schema."""
    inputs = {
        "case_id": "CASE-MANUAL-001",
        "extracted_text": "Road construction project invoice without coordinates. Start: 2025-02-01 Completion: 2026-09-30.",
        "manual_corrections": {
            "latitude": {"value": 22.7215, "provenance": "officer-supplied"},
            "longitude": {"value": 88.4812, "provenance": "officer-supplied"},
        },
    }
    mock_b = {"scene_id": "S2_B", "acquisition_date": "2025-01-20", "cloud_cover_percent": 1.0, "asset_urls": {"red": "mock_r", "nir": "mock_n", "visual": "mock_v"}}
    mock_a = {"scene_id": "S2_A", "acquisition_date": "2026-09-15", "cloud_cover_percent": 1.0, "asset_urls": {"red": "mock_r2", "nir": "mock_n2", "visual": "mock_v2"}}
    mock_change = {
        "mean_ndvi_before": 0.5, "mean_ndvi_after": 0.3, "delta_ndvi": -0.2,
        "vegetation_loss_fraction": 0.4, "potential_disturbed_area_m2": 10000.0,
        "mean_bsi_before": -0.05, "mean_bsi_after": 0.15, "delta_bsi": 0.2,
        "valid_observation_coverage_pct": 100.0,
        "rgb_before_preview_path": "mock_b.png", "rgb_after_preview_path": "mock_a.png",
    }
    with patch("agents.construction_verification.nodes.satellite_search.EarthSearchClient") as MockSTAC, \
         patch("agents.construction_verification.nodes.satellite_analysis.RasterProcessor") as MockProc:
        MockSTAC.return_value.find_best_scene.side_effect = [mock_b, mock_a]
        MockProc.return_value.analyze_change.return_value = mock_change
        result = await construction_verification_agent.ainvoke(
            inputs, config={"configurable": {"thread_id": inputs["case_id"]}}
        )

    assert result["construction_location"]["latitude"] == 22.7215
    assert result["construction_location"]["longitude"] == 88.4812
    assert result["bounding_box"] == [88.4762, 22.7165, 88.4862, 22.7265]
    assert result["construction_location"]["preliminary_search_location"] is True


@pytest.mark.asyncio
async def test_regression_5_manual_dates_update_graph_state():
    """Test 5: Manual dates update graph state and override conflicting document dates."""
    inputs = {
        "case_id": "CASE-MANUAL-DATES-001",
        "extracted_text": "Road project with obsolete dates. Start: 2023-01-01 Completion: 2023-12-31. Latitude: 22.7215 Longitude: 88.4812.",
        "manual_corrections": {
            "start_date": {"value": "2025-02-01", "provenance": "officer-supplied"},
            "completion_date": {"value": "2026-09-30", "provenance": "officer-supplied"},
        },
    }
    mock_b = {"scene_id": "S2_B", "acquisition_date": "2025-01-20", "cloud_cover_percent": 1.0, "asset_urls": {"red": "mock_r", "nir": "mock_n", "visual": "mock_v"}}
    mock_a = {"scene_id": "S2_A", "acquisition_date": "2026-09-15", "cloud_cover_percent": 1.0, "asset_urls": {"red": "mock_r2", "nir": "mock_n2", "visual": "mock_v2"}}
    mock_change = {
        "mean_ndvi_before": 0.5, "mean_ndvi_after": 0.3, "delta_ndvi": -0.2,
        "vegetation_loss_fraction": 0.4, "potential_disturbed_area_m2": 10000.0,
        "mean_bsi_before": -0.05, "mean_bsi_after": 0.15, "delta_bsi": 0.2,
        "valid_observation_coverage_pct": 100.0,
        "rgb_before_preview_path": "mock_b.png", "rgb_after_preview_path": "mock_a.png",
    }
    with patch("agents.construction_verification.nodes.satellite_search.EarthSearchClient") as MockSTAC, \
         patch("agents.construction_verification.nodes.satellite_analysis.RasterProcessor") as MockProc:
        MockSTAC.return_value.find_best_scene.side_effect = [mock_b, mock_a]
        MockProc.return_value.analyze_change.return_value = mock_change
        result = await construction_verification_agent.ainvoke(
            inputs, config={"configurable": {"thread_id": inputs["case_id"]}}
        )

    # Manual dates must override the 2023 dates present in extracted_text
    assert result["start_date"] == "2025-02-01"
    assert result["completion_date"] == "2026-09-30"


@pytest.mark.asyncio
async def test_regression_6_missing_field_validation_clears_after_corrections():
    """Test 6: Missing-field warnings clear after valid coordinates and dates are supplied."""
    inputs = {
        "case_id": "CASE-CLEAR-WARNING-001",
        "extracted_text": "Invoice text with missing information.",
        "manual_corrections": {
            "latitude": {"value": 22.7215, "provenance": "officer-supplied"},
            "longitude": {"value": 88.4812, "provenance": "officer-supplied"},
            "start_date": {"value": "2025-02-01", "provenance": "officer-supplied"},
            "completion_date": {"value": "2026-09-30", "provenance": "officer-supplied"},
        },
    }
    mock_b = {"scene_id": "S2_B", "acquisition_date": "2025-01-20", "cloud_cover_percent": 1.0, "asset_urls": {"red": "mock_r", "nir": "mock_n", "visual": "mock_v"}}
    mock_a = {"scene_id": "S2_A", "acquisition_date": "2026-09-15", "cloud_cover_percent": 1.0, "asset_urls": {"red": "mock_r2", "nir": "mock_n2", "visual": "mock_v2"}}
    mock_change = {
        "mean_ndvi_before": 0.5, "mean_ndvi_after": 0.3, "delta_ndvi": -0.2,
        "vegetation_loss_fraction": 0.4, "potential_disturbed_area_m2": 10000.0,
        "mean_bsi_before": -0.05, "mean_bsi_after": 0.15, "delta_bsi": 0.2,
        "valid_observation_coverage_pct": 100.0,
        "rgb_before_preview_path": "mock_b.png", "rgb_after_preview_path": "mock_a.png",
    }
    with patch("agents.construction_verification.nodes.satellite_search.EarthSearchClient") as MockSTAC, \
         patch("agents.construction_verification.nodes.satellite_analysis.RasterProcessor") as MockProc:
        MockSTAC.return_value.find_best_scene.side_effect = [mock_b, mock_a]
        MockProc.return_value.analyze_change.return_value = mock_change
        result = await construction_verification_agent.ainvoke(
            inputs, config={"configurable": {"thread_id": inputs["case_id"]}}
        )

    # Errors should not contain missing fields
    missing_errors = [e for e in result.get("errors", []) if "Missing required fields" in e]
    assert len(missing_errors) == 0


@pytest.mark.asyncio
async def test_regression_7_chat_messages_retain_case_context():
    """Test 7: Chat messages do not wipe existing case context and parse inline updates."""
    # First invoke with base case info
    initial_state = {
        "case_id": "CASE-CHAT-001",
        "project_name": "Rural Link Road, Barasat–Kazipara",
        "extracted_text": ROAD_LETTER_TEXT,
        "start_date": "2025-02-01",
        "completion_date": "2026-09-30",
        "messages": [HumanMessage(content="use coordinates lat 22.7215, lon 88.4812")],
    }
    result = await construction_verification_agent.ainvoke(initial_state)

    # State retains project name and updates coordinates from chat
    assert "Barasat–Kazipara" in result.get("project_name", "")
    assert result["construction_location"]["latitude"] == 22.7215
    assert result["construction_location"]["longitude"] == 88.4812
    ai_msg = result["messages"][-1].content
    assert "22.7215" in ai_msg
    assert "88.4812" in ai_msg


@pytest.mark.asyncio
async def test_regression_8_repeated_updates_preserve_state():
    """Test 8: Repeated state updates preserve previous fields without overwriting with empty defaults."""
    case_id = "CASE-REPEAT-001"
    rec1 = load_case_record(case_id)
    rec1["project_name"] = "Barasat Road Project"
    rec1["extracted_text"] = "Some initial text"
    save_case_record(case_id, rec1)

    # Simulate second partial update: only coordinates
    rec2 = load_case_record(case_id)
    rec2["construction_location"] = {"latitude": 22.7215, "longitude": 88.4812}
    save_case_record(case_id, rec2)

    loaded = load_case_record(case_id)
    assert loaded["project_name"] == "Barasat Road Project"
    assert loaded["extracted_text"] == "Some initial text"
    assert loaded["construction_location"]["latitude"] == 22.7215


@pytest.mark.asyncio
async def test_regression_9_case_isolation():
    """Test 9: A second case cannot access or inherit a first case's document or state."""
    case_a = "CASE-ALPHA-001"
    case_b = "CASE-BETA-002"

    rec_a = load_case_record(case_a)
    rec_a["project_name"] = "Alpha Road Project"
    rec_a["document_paths"] = ["/path/to/alpha.pdf"]
    save_case_record(case_a, rec_a)

    rec_b = load_case_record(case_b)
    # Case B should start empty
    assert rec_b["project_name"] == ""
    assert len(rec_b["document_paths"]) == 0
    assert "Alpha Road Project" not in rec_b["project_name"]


from agents.construction_verification.schemas import (
    VisionAnalysisOutput,
    VisionClaimFinding,
    VisionExecutionAudit,
)


@pytest.mark.asyncio
async def test_regression_10_satellite_verification_discovery_or_error():
    """Test 10: Satellite verification starts after valid coordinates and dates are supplied (Gemini success)."""
    inputs = {
        "case_id": "CASE-SAT-001",
        "project_name": "Rural Link Road, Barasat–Kazipara",
        "construction_type": "Road construction",
        "bounding_box": [88.4762, 22.7165, 88.4862, 22.7265],
        "start_date": "2025-02-01",
        "completion_date": "2026-09-30",
        "construction_claims": [
            {"activity": "Bituminous road construction", "claimed_status": "completed"}
        ],
    }

    mock_audit = VisionExecutionAudit(
        vision_actually_executed=True,
        parsed_response_status="SUCCESS",
        model_used="gemini-3.8-flash",
    )
    mock_vision = VisionAnalysisOutput(
        findings=[
            VisionClaimFinding(
                claim="Bituminous road construction",
                supports_claim=True,
                reasoning="Spectral change and high BSI confirm road corridor construction.",
            )
        ],
        vision_audit=mock_audit,
    )

    with patch("agents.construction_verification.nodes.satellite_search.EarthSearchClient") as MockSTAC:
        instance = MockSTAC.return_value
        instance.find_best_scene.side_effect = [
            {"scene_id": "S2A_BEFORE", "acquisition_date": "2025-01-20", "cloud_cover_percent": 2.5, "asset_urls": {"red": "mock_r", "nir": "mock_n", "visual": "mock_v"}},
            {"scene_id": "S2A_AFTER", "acquisition_date": "2026-09-15", "cloud_cover_percent": 4.1, "asset_urls": {"red": "mock_r2", "nir": "mock_n2", "visual": "mock_v2"}},
        ]
        with patch("agents.construction_verification.nodes.satellite_analysis.RasterProcessor") as MockProc:
            proc_inst = MockProc.return_value
            proc_inst.analyze_change.return_value = {
                "mean_ndvi_before": 0.52,
                "mean_ndvi_after": 0.31,
                "delta_ndvi": -0.21,
                "vegetation_loss_fraction": 0.38,
                "potential_disturbed_area_m2": 11500.0,
                "mean_bsi_before": -0.03,
                "mean_bsi_after": 0.15,
                "delta_bsi": 0.18,
                "valid_observation_coverage_pct": 100.0,
                "rgb_before_preview_path": "mock_b.png",
                "rgb_after_preview_path": "mock_a.png",
            }
            with patch("agents.construction_verification.nodes.vision_analysis.GeminiVerificationClient.analyze_satellite_imagery", return_value=mock_vision), \
                 patch("agents.construction_verification.nodes.vision_analysis.Path.exists", return_value=True):
                result = await construction_verification_agent.ainvoke(
                    inputs, config={"configurable": {"thread_id": inputs["case_id"]}}
                )
                assert "satellite_metrics" in result
                assert result["satellite_metrics"]["delta_ndvi"] == -0.21
                assert "verification_results" in result
                assert result["verification_results"][0]["status"] == "SUPPORTED"
                assert result["vision_audit"]["vision_actually_executed"] is True
                assert result["vision_audit"]["model_used"] == "gemini-3.8-flash"


@pytest.mark.asyncio
async def test_regression_10_when_gemini_fails():
    """Test 10 (Gemini failure): Assert what is asserted when Gemini fails (NOT_ANALYZED status)."""
    inputs = {
        "case_id": "CASE-SAT-FAIL-001",
        "project_name": "Rural Link Road, Barasat–Kazipara",
        "construction_type": "Road construction",
        "bounding_box": [88.4762, 22.7165, 88.4862, 22.7265],
        "start_date": "2025-02-01",
        "completion_date": "2026-09-30",
        "construction_claims": [
            {"activity": "Bituminous road construction", "claimed_status": "completed"}
        ],
    }

    mock_audit = VisionExecutionAudit(
        vision_actually_executed=False,
        parsed_response_status="QUOTA_EXHAUSTED",
        error_message="Gemini API Quota Exhausted (HTTP 429)",
    )
    mock_failed_vision = VisionAnalysisOutput(
        findings=[
            VisionClaimFinding(
                claim="Bituminous road construction",
                supports_claim="NOT_ANALYZED",
                reasoning="QUOTA_EXHAUSTED: Gemini API rate limit reached (HTTP 429).",
            )
        ],
        vision_audit=mock_audit,
    )

    with patch("agents.construction_verification.nodes.satellite_search.EarthSearchClient") as MockSTAC:
        instance = MockSTAC.return_value
        instance.find_best_scene.side_effect = [
            {"scene_id": "S2A_BEFORE", "acquisition_date": "2025-01-20", "cloud_cover_percent": 2.5, "asset_urls": {"red": "mock_r", "nir": "mock_n", "visual": "mock_v"}},
            {"scene_id": "S2A_AFTER", "acquisition_date": "2026-09-15", "cloud_cover_percent": 4.1, "asset_urls": {"red": "mock_r2", "nir": "mock_n2", "visual": "mock_v2"}},
        ]
        with patch("agents.construction_verification.nodes.satellite_analysis.RasterProcessor") as MockProc:
            proc_inst = MockProc.return_value
            proc_inst.analyze_change.return_value = {
                "mean_ndvi_before": 0.52, "mean_ndvi_after": 0.31, "delta_ndvi": -0.21,
                "vegetation_loss_fraction": 0.38, "potential_disturbed_area_m2": 11500.0,
                "mean_bsi_before": -0.03, "mean_bsi_after": 0.15, "delta_bsi": 0.18,
                "valid_observation_coverage_pct": 100.0,
                "rgb_before_preview_path": "mock_b.png", "rgb_after_preview_path": "mock_a.png",
            }
            with patch("agents.construction_verification.nodes.vision_analysis.GeminiVerificationClient.analyze_satellite_imagery", return_value=mock_failed_vision), \
                 patch("agents.construction_verification.nodes.vision_analysis.Path.exists", return_value=True):
                result = await construction_verification_agent.ainvoke(
                    inputs, config={"configurable": {"thread_id": inputs["case_id"]}}
                )
                assert result["verification_results"][0]["status"] == "NOT_ANALYZED"
                assert result["overall_status"] == "NOT_ANALYZED_REQUIRES_INSPECTION"
                assert result["vision_audit"]["vision_actually_executed"] is False
                assert result["vision_audit"]["parsed_response_status"] == "QUOTA_EXHAUSTED"

