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


@pytest.mark.asyncio
async def test_construction_agent_invoke():
    """Test 17e: End-to-end invocation of construction-verification LangGraph agent."""
    input_data = {
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

    result = await construction_verification_agent.ainvoke(input_data)
    assert "verification_results" in result
    assert len(result["verification_results"]) == 2
    assert "report" in result
    assert "messages" in result
    assert len(result["messages"]) > 0

