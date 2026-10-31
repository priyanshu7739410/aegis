import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage

from agents.agents import get_agent, get_all_agent_info
from agents.construction_verification import construction_verification_agent
from service.service import app


def test_agent_registration():
    """Test 16: Verify construction-verification agent is registered in the registry."""
    agent = get_agent("construction-verification")
    assert agent is not None
    assert agent == construction_verification_agent

    all_agents = get_all_agent_info()
    keys = [a.key for a in all_agents]
    assert "construction-verification" in keys


@pytest.mark.asyncio
async def test_fastapi_endpoints_integration(tmp_path):
    """Test 17: Verify FastAPI endpoints for /info, /construction/upload, /construction/decision, and /case."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Test 17a: /info lists construction-verification agent
        info_resp = await ac.get("/info")
        assert info_resp.status_code == 200
        info_data = info_resp.json()
        agent_keys = [a["key"] for a in info_data["agents"]]
        assert "construction-verification" in agent_keys

        # Test 17b: /construction/upload
        test_file_content = (
            b"CONTRACT REPORT: NH-44 Expansion. Latitude: 12.9716 Longitude: 77.5946. "
            b"Start Date: 2024-01-10 Completion Date: 2024-04-15. "
            b"Claim 1: Vegetation clearing completed."
        )
        files = {"file": ("claim_doc.pdf", test_file_content, "application/pdf")}
        upload_resp = await ac.post("/construction/upload", files=files, data={"run_ocr": "false"})
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        assert "file_path" in upload_data
        assert upload_data["file_name"] == "claim_doc.pdf"

        # Test 17c: /construction/decision records official decision
        decision_payload = {
            "case_id": "CASE-INT-TEST-001",
            "decision": "APPROVE",
            "officer_name_or_id": "INSPECTOR-JOHN",
            "notes": "Verified against all satellite spectral indices.",
        }
        dec_resp = await ac.post("/construction/decision", json=decision_payload)
        assert dec_resp.status_code == 200
        dec_data = dec_resp.json()
        assert dec_data["status"] == "RECORDED"
        assert dec_data["decision_record"]["decision"] == "APPROVE"

        # Test 17d: /construction/case/{case_id} retrieves recorded decision
        case_resp = await ac.get("/construction/case/CASE-INT-TEST-001")
        assert case_resp.status_code == 200
        case_data = case_resp.json()
        assert case_data["case_id"] == "CASE-INT-TEST-001"
        assert case_data["officer_name_or_id"] == "INSPECTOR-JOHN"


from unittest.mock import patch
from agents.construction_verification.schemas import (
    VisionAnalysisOutput,
    VisionClaimFinding,
    VisionExecutionAudit,
)


@pytest.mark.asyncio
async def test_construction_agent_invoke():
    """Test 17e: End-to-end invocation of construction-verification LangGraph agent with mocked external APIs."""
    input_data = {
        "case_id": "CASE-E2E-001",
        "extracted_text": (
            "CONTRACT REPORT\n"
            "Project: State Highway Site 4\n"
            "Construction Type: Highway Expansion\n"
            "Location: Latitude 12.9716, Longitude 77.5946\n"
            "Start Date: 2024-01-10\n"
            "Claimed Completion Date: 2024-04-15\n"
            "Claim 1: Complete vegetation clearing (10,000 m2) - completed.\n"
            "Claim 2: Deep foundation concrete piling - completed.\n"
        ),
        "messages": [HumanMessage(content="Please verify the construction claims in this document.")],
    }

    mock_audit = VisionExecutionAudit(
        vision_actually_executed=True,
        parsed_response_status="SUCCESS",
        model_used="gemini-3.8-flash",
        gemini_model_id="gemini-3.8-flash",
    )
    mock_vision_out = VisionAnalysisOutput(
        findings=[
            VisionClaimFinding(
                claim="Complete vegetation clearing (10,000 m2)",
                supports_claim=True,
                reasoning="Spectral loss aligns with claimed clearing.",
            ),
            VisionClaimFinding(
                claim="Deep foundation concrete piling",
                supports_claim=None,
                reasoning="Sub-surface piling not visible from 10m optical satellite.",
            ),
        ],
        vision_audit=mock_audit,
    )

    with patch("agents.construction_verification.nodes.satellite_search.EarthSearchClient") as MockSTAC, \
         patch("agents.construction_verification.nodes.satellite_analysis.RasterProcessor") as MockProc, \
         patch("agents.construction_verification.nodes.vision_analysis.GeminiVerificationClient.analyze_satellite_imagery", return_value=mock_vision_out):
        
        stac_inst = MockSTAC.return_value
        stac_inst.find_best_scene.side_effect = [
            {"scene_id": "S2_B", "acquisition_date": "2024-01-05", "cloud_cover_percent": 1.0, "asset_urls": {"red": "mock_r", "nir": "mock_n", "visual": "mock_v"}},
            {"scene_id": "S2_A", "acquisition_date": "2024-04-20", "cloud_cover_percent": 2.0, "asset_urls": {"red": "mock_r", "nir": "mock_n", "visual": "mock_v"}},
        ]
        proc_inst = MockProc.return_value
        proc_inst.analyze_change.return_value = {
            "mean_ndvi_before": 0.6, "mean_ndvi_after": 0.2, "delta_ndvi": -0.4,
            "vegetation_loss_fraction": 0.7, "potential_disturbed_area_m2": 10000.0,
            "mean_bsi_before": -0.1, "mean_bsi_after": 0.2, "delta_bsi": 0.3,
            "valid_observation_coverage_pct": 100.0,
            "rgb_before_preview_path": "mock_b.png", "rgb_after_preview_path": "mock_a.png",
        }

        result = await construction_verification_agent.ainvoke(
            input_data, config={"configurable": {"thread_id": "thread-e2e-invoke"}}
        )

    # Tight assertions when Gemini vision executes successfully
    assert "verification_results" in result
    assert len(result["verification_results"]) == 2
    assert result["verification_results"][0]["status"] == "SUPPORTED"
    assert result["verification_results"][1]["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["vision_audit"]["vision_actually_executed"] is True
    assert result["vision_audit"]["model_used"] == "gemini-3.8-flash"
    assert "report" in result
    assert "messages" in result
    assert len(result["messages"]) > 0


@pytest.mark.asyncio
async def test_construction_agent_invoke_when_gemini_fails():
    """Test 17f (Item 10): State what is asserted when Gemini fails and verify NOT_ANALYZED status."""
    input_data = {
        "case_id": "CASE-E2E-FAIL-001",
        "extracted_text": (
            "CONTRACT REPORT\n"
            "Project: State Highway Site 4\n"
            "Construction Type: Highway Expansion\n"
            "Location: Latitude 12.9716, Longitude 77.5946\n"
            "Start Date: 2024-01-10\n"
            "Claimed Completion Date: 2024-04-15\n"
            "Claim 1: Complete vegetation clearing (10,000 m2) - completed.\n"
        ),
        "messages": [HumanMessage(content="Please verify.")],
    }

    mock_failed_audit = VisionExecutionAudit(
        vision_actually_executed=False,
        parsed_response_status="QUOTA_EXHAUSTED",
        error_message="Gemini API Quota Exhausted (HTTP 429)",
    )
    mock_failed_vision = VisionAnalysisOutput(
        findings=[
            VisionClaimFinding(
                claim="Complete vegetation clearing (10,000 m2)",
                supports_claim="NOT_ANALYZED",
                reasoning="QUOTA_EXHAUSTED: Gemini API rate limit reached (HTTP 429).",
            )
        ],
        vision_audit=mock_failed_audit,
    )

    with patch("agents.construction_verification.nodes.satellite_search.EarthSearchClient") as MockSTAC, \
         patch("agents.construction_verification.nodes.satellite_analysis.RasterProcessor") as MockProc, \
         patch("agents.construction_verification.nodes.vision_analysis.GeminiVerificationClient.analyze_satellite_imagery", return_value=mock_failed_vision):
        
        stac_inst = MockSTAC.return_value
        stac_inst.find_best_scene.side_effect = [
            {"scene_id": "S2_B", "acquisition_date": "2024-01-05", "cloud_cover_percent": 1.0, "asset_urls": {"red": "mock_r", "nir": "mock_n", "visual": "mock_v"}},
            {"scene_id": "S2_A", "acquisition_date": "2024-04-20", "cloud_cover_percent": 2.0, "asset_urls": {"red": "mock_r", "nir": "mock_n", "visual": "mock_v"}},
        ]
        proc_inst = MockProc.return_value
        proc_inst.analyze_change.return_value = {
            "mean_ndvi_before": 0.6, "mean_ndvi_after": 0.2, "delta_ndvi": -0.4,
            "vegetation_loss_fraction": 0.7, "potential_disturbed_area_m2": 10000.0,
            "mean_bsi_before": -0.1, "mean_bsi_after": 0.2, "delta_bsi": 0.3,
            "valid_observation_coverage_pct": 100.0,
            "rgb_before_preview_path": "mock_b.png", "rgb_after_preview_path": "mock_a.png",
        }

        result = await construction_verification_agent.ainvoke(
            input_data, config={"configurable": {"thread_id": "thread-e2e-fail"}}
        )

    # Tight assertions when Gemini fails:
    # 1. Claim status MUST be VISION_NOT_EXECUTED (never INSUFFICIENT_EVIDENCE or VERIFIED)
    assert result["verification_results"][0]["status"] == "VISION_NOT_EXECUTED"
    # 2. Overall status must require inspection due to unanalyzed status
    assert result["overall_status"] == "VISION_NOT_EXECUTED_REQUIRES_INSPECTION"
    # 3. Vision audit must explicitly record non-execution
    assert result["vision_audit"]["vision_actually_executed"] is False
    assert result["vision_audit"]["parsed_response_status"] in ("QUOTA_EXHAUSTED", "VISION_NOT_EXECUTED")

