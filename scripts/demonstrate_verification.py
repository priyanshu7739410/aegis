"""Demonstration script for AI-Powered Construction Verification & Final Payment Approval System."""

import asyncio
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from agents.agents import get_agent
from agents.construction_verification.schemas import OfficerDecisionRecord, OfficerDecisionType
from service.construction import record_officer_decision, CaseDecisionInput


async def run_demonstration():
    print("=" * 70)
    print("AI-POWERED CONSTRUCTION VERIFICATION & PAYMENT APPROVAL SYSTEM")
    print("DEMONSTRATION RUN")
    print("=" * 70)

    # 1. Retrieve Registered LangGraph Agent
    agent = get_agent("construction-verification")
    print("\n[STEP 1] Agent Retrieved from Service Registry:")
    print(f"  Agent ID: 'construction-verification'")
    print(f"  Graph Type: {type(agent)}")

    # 2. Synthetic Contractor Claim Document
    synthetic_doc_text = """
    CONTRACTOR COMPLETION REPORT & PAYMENT APPLICATION
    Project: NH-44 Highway Expansion & Interchange Site B
    Construction Type: Highway Road Construction
    Location: Latitude 12.9716, Longitude 77.5946 (WGS84 Coordinates)
    Period of Performance:
      - Project Start Date: 2024-01-10
      - Claimed Completion Date: 2024-04-15
    Contractor: ABC Infra Infrastructure Ltd.
    Invoice Value: $450,000 (Final Milestone)

    Claimed Completed Activities:
    1. Complete vegetation clearing and topsoil stripping (12,000 sq.m) - completed.
    2. Subgrade excavation, grading, and soil compaction - completed.
    3. Structural reinforced concrete box culvert installation - completed.
    """

    print("\n[STEP 2] Contractor Document Received:")
    print("  Document: Invoice / Completion Application")
    print(f"  Size: {len(synthetic_doc_text)} characters")

    # 3. Execute LangGraph Workflow (OCR -> Extraction -> STAC -> Raster -> Vision -> Report)
    print("\n[STEP 3] Executing LangGraph Orchestrated Workflow...")
    input_state = {
        "case_id": "DEMO-CASE-2024-001",
        "extracted_text": synthetic_doc_text,
        "messages": [HumanMessage(content="Verify contractor claims for final payment approval.")],
    }

    result = await agent.ainvoke(input_state)

    print("\n[STEP 4] Multi-Temporal Satellite & Spectral Analysis Results:")
    metrics = result.get("satellite_metrics", {})
    print(f"  Before NDVI (Mean): {metrics.get('mean_ndvi_before'):.3f}")
    print(f"  After NDVI (Mean):  {metrics.get('mean_ndvi_after'):.3f}")
    print(f"  Delta NDVI:         {metrics.get('delta_ndvi'):+.3f}")
    print(f"  Vegetation Loss:    {metrics.get('vegetation_loss_fraction', 0)*100:.1f}%")
    print(f"  Disturbed Area:     {metrics.get('potential_disturbed_area_m2'):,.0f} m²")
    print(f"  Bare Soil (BSI):    {metrics.get('mean_bsi_after'):.3f} (Delta {metrics.get('delta_bsi'):+.3f})")
    print(f"  Site Cloud-Free:    {metrics.get('valid_observation_coverage_pct'):.1f}%")

    print("\n[STEP 5] Claim-by-Claim AI Verification:")
    verification_results = result.get("verification_results", [])
    for idx, r in enumerate(verification_results, 1):
        st = r["status"]
        icon = "🟢" if st == "SUPPORTED" else "🔴" if st == "NOT_SUPPORTED" else "🟡"
        print(f"\n  Claim {idx}: '{r['claim']}'")
        print(f"    Status: {icon} {st}")
        print(f"    Reason: {r['reason']}")
        print(f"    Limitations: {', '.join(r.get('limitations', []))}")
        print(f"    Recommended Action: {r['recommended_next_action']}")

    overall_status = result.get("overall_status")
    print(f"\n  Overall Preliminary Case Status: {overall_status}")
    print(f"  Drone Inspection Required: {result.get('requires_drone_verification')}")

    # 4. Drone Verification Step (supplementing inconclusive sub-surface claim)
    print("\n[STEP 6] Simulating On-Site Drone Inspection Upload...")
    drone_input = {
        "case_id": "DEMO-CASE-2024-001",
        "construction_claims": result.get("construction_claims", []),
        "verification_results": verification_results,
        "report": result.get("report", {}),
        "drone_image_paths": ["outputs/construction/demo_drone_culvert.jpg"],
    }
    updated_result = await agent.ainvoke(drone_input)

    print("\n[STEP 7] Updated Verification Findings After Drone Evidence:")
    for idx, r in enumerate(updated_result.get("verification_results", []), 1):
        st = r["status"]
        icon = "🟢" if st == "SUPPORTED" else "🔴" if st == "NOT_SUPPORTED" else "🟡"
        print(f"  Claim {idx}: {icon} {st} - '{r['claim']}'")
        print(f"    Updated Reason: {r['reason']}")

    print(f"\n  Final Evidence Status: {updated_result.get('overall_status')}")

    # 5. Human Officer Approval Step
    print("\n[STEP 8] Official Human Officer Payment Approval:")
    print("  *SECURITY COMPLIANCE*: AI cannot approve payment. Recording human officer decision...")
    decision_input = CaseDecisionInput(
        case_id="DEMO-CASE-2024-001",
        decision=OfficerDecisionType.APPROVE,
        officer_name_or_id="OFFICER-RAJESH-KUMAR",
        notes="Verified against Sentinel-2 vegetation clearing metrics and high-resolution drone orthophoto.",
    )
    decision_resp = await record_officer_decision(decision_input)
    print(f"  Decision Status: {decision_resp.status}")
    print(f"  Officer: {decision_resp.decision_record.officer_name_or_id}")
    print(f"  Action: {decision_resp.decision_record.decision}")
    print(f"  Timestamp: {decision_resp.decision_record.timestamp}")
    print(f"  Notes: {decision_resp.decision_record.notes}")
    print(f"  Audit Receipt: {decision_resp.message}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_demonstration())
