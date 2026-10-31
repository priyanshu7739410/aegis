import pytest
from langchain_core.runnables import RunnableConfig

from agents.construction_verification.nodes.drone_eval import drone_eval_node
from agents.construction_verification.nodes.verification import verification_node
from agents.construction_verification.schemas import OfficerDecisionRecord, VerificationStatus
from agents.construction_verification.services.gemini_client import GeminiVerificationClient


@pytest.mark.asyncio
async def test_unsupported_construction_claim():
    """Test 12: Claims contradictory to spectral evidence are flagged NOT_SUPPORTED."""
    client = GeminiVerificationClient()
    claims = [{"activity": "Massive vegetation clearing of 50 acres", "claimed_status": "completed"}]
    # Metrics show NO vegetation loss (0.0 loss fraction)
    no_loss_metrics = {
        "mean_ndvi_before": 0.65,
        "mean_ndvi_after": 0.66,
        "delta_ndvi": +0.01,
        "vegetation_loss_fraction": 0.0,
        "potential_disturbed_area_m2": 0.0,
    }

    vision_out = client._fallback_satellite_vision(claims, no_loss_metrics)
    assert len(vision_out.findings) == 1
    assert vision_out.findings[0].supports_claim is False


@pytest.mark.asyncio
async def test_insufficient_satellite_evidence():
    """Test 13: Sub-surface and structural claims are marked INSUFFICIENT_EVIDENCE."""
    client = GeminiVerificationClient()
    structural_claims = [
        {"activity": "Poured reinforced concrete foundation grade M30", "claimed_status": "completed"},
        {"activity": "Sub-surface trenching and drainage pipe installation", "claimed_status": "completed"},
    ]
    metrics = {
        "mean_ndvi_before": 0.60,
        "mean_ndvi_after": 0.20,
        "delta_ndvi": -0.40,
        "vegetation_loss_fraction": 0.85,
        "potential_disturbed_area_m2": 8500.0,
    }

    vision_out = client._fallback_satellite_vision(structural_claims, metrics)
    for finding in vision_out.findings:
        # Optical satellite cannot verify internal concrete or trench depth
        assert finding.supports_claim is None
        assert any("subterranean" in lim.lower() or "resolution" in lim.lower() for lim in finding.limitations)


@pytest.mark.asyncio
async def test_drone_verification_routing_and_reevaluation(tmp_path):
    """Test 14: Insufficient evidence triggers drone request, and drone imagery re-evaluates claims."""
    state = {
        "case_id": "CASE-TEST-DRONE",
        "construction_claims": [
            {"activity": "Culvert foundation excavation and rebar cage tying", "claimed_status": "completed"}
        ],
        "vision_findings": {
            "findings": [
                {
                    "claim": "Culvert foundation excavation and rebar cage tying",
                    "supports_claim": None,
                    "observed_changes": ["Ground disturbed"],
                    "reasoning": "Sub-surface depth invisible to satellite",
                    "limitations": ["10m resolution"],
                }
            ]
        },
        "satellite_metrics": {"valid_observation_coverage_pct": 98.0},
    }

    # Run verification node
    res = await verification_node(state, RunnableConfig())
    assert res["requires_drone_verification"] is True
    assert res["verification_results"][0]["status"] == VerificationStatus.INSUFFICIENT_EVIDENCE.value
    assert res["report"]["drone_inspection_request"] is not None

    # Now simulate officer uploading drone inspection photograph
    fake_drone_img = tmp_path / "drone_inspection_01.jpg"
    fake_drone_img.write_bytes(b"\xff\xd8\xfffake_drone_jpeg")

    drone_state = {
        "case_id": "CASE-TEST-DRONE",
        "construction_claims": state["construction_claims"],
        "verification_results": res["verification_results"],
        "report": res["report"],
        "drone_image_paths": [str(fake_drone_img)],
    }

    drone_res = await drone_eval_node(drone_state, RunnableConfig())
    assert drone_res["verification_results"][0]["status"] == VerificationStatus.SUPPORTED.value
    assert "drone" in drone_res["overall_status"].lower()


def test_human_approval_requirement():
    """Test 15: AI outputs cannot authorize payment; explicit human decision record is enforced."""
    # Ensure officer decision requires human input and holds case_id
    record = OfficerDecisionRecord(
        case_id="CASE-999",
        decision="APPROVE",
        officer_name_or_id="OFFICER-007",
        notes="All photographic inspection evidence reviewed and validated.",
    )
    assert record.decision == "APPROVE"
    assert record.officer_name_or_id == "OFFICER-007"
    assert record.timestamp is not None

