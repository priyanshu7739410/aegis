from datetime import datetime, timezone
import logging
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.construction_verification.schemas import (
    ClaimVerificationResult,
    VerificationStatus,
)
from agents.construction_verification.services.gemini_client import GeminiVerificationClient
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)


async def drone_eval_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict[str, Any]:
    """Re-evaluate claims flagged as insufficient using uploaded drone/site inspection images."""
    drone_images = state.get("drone_image_paths", [])
    claims = state.get("construction_claims", [])
    existing_results = list(state.get("verification_results", []))
    report = dict(state.get("report", {}))
    case_id = state.get("case_id", "CASE-001")

    if not drone_images:
        logger.info("No drone/site inspection images uploaded for re-evaluation.")
        return {}

    # Identify claims that were previously INSUFFICIENT_EVIDENCE
    insufficient_claims = [
        c for c in claims
        if any(r["claim"] == c.get("activity") and r["status"] == VerificationStatus.INSUFFICIENT_EVIDENCE.value for r in existing_results)
    ]

    target_claims = insufficient_claims if insufficient_claims else claims

    gemini_client = GeminiVerificationClient()
    drone_findings = await gemini_client.analyze_drone_imagery(
        claims=target_claims,
        drone_image_paths=drone_images,
    )

    drone_findings_dicts = [f.model_dump() for f in drone_findings]

    # Update verification results with drone evidence
    updated_results: list[dict[str, Any]] = []
    for r in existing_results:
        matching_finding = next((df for df in drone_findings_dicts if df["claim"] == r["claim"]), None)
        if matching_finding and matching_finding.get("supports_claim"):
            new_r = ClaimVerificationResult(
                claim=r["claim"],
                status=VerificationStatus.SUPPORTED.value,
                observations=r.get("observations", []) + matching_finding.get("observations", []),
                reason=f"Satellite observation was supplemented by drone inspection: {matching_finding.get('reasoning')}",
                limitations=matching_finding.get("limitations", []),
                recommended_next_action="Drone inspection verified visible physical work. Officer may consider payment approval.",
            )
            updated_results.append(new_r.model_dump())
        else:
            updated_results.append(r)

    # Check new overall status
    if any(r["status"] == VerificationStatus.NOT_SUPPORTED.value for r in updated_results):
        overall_status = "POTENTIAL_CONTRADICTION"
    elif any(r["status"] == VerificationStatus.INSUFFICIENT_EVIDENCE.value for r in updated_results):
        overall_status = "PARTIALLY_VERIFIED_FURTHER_INSPECTION_NEEDED"
    else:
        overall_status = "VERIFIED_BY_SATELLITE_AND_DRONE_INSPECTION"

    # Update report
    report["overall_status"] = overall_status
    report["claims_summary"] = updated_results
    report["drone_findings"] = drone_findings_dicts
    report.setdefault("audit_log", []).append(
        f"Drone inspection re-evaluation completed at {datetime.now(timezone.utc).isoformat()} with {len(drone_images)} image(s)."
    )

    msg_lines = [
        f"### Drone Inspection Verification Complete — Case `{case_id}`",
        f"Evaluated {len(drone_images)} high-resolution inspection image(s).",
        f"**New Status**: `{overall_status}`",
        "",
        "| Claim | Updated Status | Evidence Source |",
        "| :--- | :--- | :--- |",
    ]
    for r in updated_results:
        st_icon = "🟢" if r["status"] == "SUPPORTED" else "🔴" if r["status"] == "NOT_SUPPORTED" else "🟡"
        msg_lines.append(f"| **{r['claim']}** | {st_icon} {r['status']} | Satellite + Drone Orthophoto |")

    msg_lines.append("")
    msg_lines.append("**Officer Decision**: Please review the updated evidence and record your final decision.")

    return {
        "verification_results": updated_results,
        "overall_status": overall_status,
        "drone_findings": drone_findings_dicts,
        "report": report,
        "requires_drone_verification": any(r["status"] == VerificationStatus.INSUFFICIENT_EVIDENCE.value for r in updated_results),
        "messages": [AIMessage(content="\n".join(msg_lines))],
    }
