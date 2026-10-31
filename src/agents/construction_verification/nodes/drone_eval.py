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

    # Identify claims that were previously INSUFFICIENT_EVIDENCE or NOT_ANALYZED
    unresolved_claims = [
        c for c in claims
        if any(r["claim"] == c.get("activity") and r["status"] in (VerificationStatus.INSUFFICIENT_EVIDENCE.value, VerificationStatus.NOT_ANALYZED.value) for r in existing_results)
    ]

    target_claims = unresolved_claims if unresolved_claims else claims

    gemini_client = GeminiVerificationClient()
    drone_findings = await gemini_client.analyze_drone_imagery(
        claims=target_claims,
        drone_image_paths=drone_images,
    )

    raw_list = drone_findings.get("findings", []) if isinstance(drone_findings, dict) else drone_findings
    drone_findings_dicts = [f.model_dump() if hasattr(f, "model_dump") else f for f in raw_list]

    # Update verification results with drone evidence
    updated_results: list[dict[str, Any]] = []
    for r in existing_results:
        matching_finding = next((df for df in drone_findings_dicts if df["claim"] == r["claim"]), None)
        if matching_finding:
            sup = matching_finding.get("supports_claim")
            if sup in ("SUPPORTED", "supported", True):
                st_val = VerificationStatus.SUPPORTED.value
                rec_act = "Photographic inspection verified physical work. Officer may consider payment approval."
            elif sup in ("NOT_SUPPORTED", "not_supported", False):
                st_val = VerificationStatus.NOT_SUPPORTED.value
                rec_act = "Photographic inspection contradicted claim. Recommend halting payment or requesting contractor clarification."
            elif sup in ("VISION_NOT_EXECUTED",):
                st_val = VerificationStatus.VISION_NOT_EXECUTED.value
                rec_act = "Inspection vision analysis could not be executed. Upload valid photos or perform manual audit."
            else:
                st_val = VerificationStatus.INSUFFICIENT_EVIDENCE.value
                rec_act = "Inspection photograph remains inconclusive for this specific claim. Physical test certificate required."

            obs_drone = matching_finding.get("observations")
            extra_obs = obs_drone if isinstance(obs_drone, list) else ([obs_drone] if obs_drone else [])

            new_r = ClaimVerificationResult(
                claim=r["claim"],
                status=st_val,
                observations=list(r.get("observations") or []) + extra_obs,
                reason=f"Satellite observation supplemented by photo inspection: {matching_finding.get('reasoning') or matching_finding.get('observations')}",
                limitations=matching_finding.get("limitations", []),
                recommended_next_action=rec_act,
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
        "requires_drone_verification": any(r["status"] in (VerificationStatus.INSUFFICIENT_EVIDENCE.value, VerificationStatus.NOT_ANALYZED.value) for r in updated_results),
        "messages": [AIMessage(content="\n".join(msg_lines))],
    }
