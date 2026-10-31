from datetime import datetime, timezone
import logging
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.construction_verification.schemas import (
    ClaimVerificationResult,
    SatelliteMetrics,
    VerificationReport,
    VerificationStatus,
)
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)


async def verification_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict[str, Any]:
    """Synthesize satellite metrics and vision findings into claim-by-claim verification report."""
    claims = state.get("construction_claims", [])
    vision_findings = state.get("vision_findings", {}).get("findings", [])
    metrics = state.get("satellite_metrics", {})
    case_id = state.get("case_id", "CASE-001")
    project_name = state.get("project_name", "Construction Project")
    construction_type = state.get("construction_type", "Civil Works")
    location = state.get("construction_location", {})
    start_date = state.get("start_date")
    completion_date = state.get("completion_date")

    verification_results: list[dict[str, Any]] = []
    requires_drone = False

    # Match each claim with vision finding & deterministic metrics
    for claim_dict in claims:
        activity = claim_dict.get("activity", "")
        # Find matching vision finding
        finding = next((f for f in vision_findings if f.get("claim") == activity), None)

        if finding:
            supports = finding.get("supports_claim")
            obs = finding.get("observed_changes", [])
            reason = finding.get("reasoning", "")
            limits = finding.get("limitations", [])

            if supports is True:
                status = VerificationStatus.SUPPORTED
                action = "Claim supported by visible satellite observation. Officer review recommended."
            elif supports is False:
                status = VerificationStatus.NOT_SUPPORTED
                action = "Reject or demand contractor explanation due to contradictory satellite evidence."
                requires_drone = True
            else:
                status = VerificationStatus.INSUFFICIENT_EVIDENCE
                action = "Request high-resolution drone imagery or on-site physical engineering inspection."
                requires_drone = True
        else:
            # Default fallback
            status = VerificationStatus.INSUFFICIENT_EVIDENCE
            obs = ["No conclusive satellite observation available for this activity."]
            reason = "Satellite resolution or observation conditions cannot establish completion of this claim."
            limits = ["Sentinel-2 spatial resolution (10m) cannot resolve fine-grained or sub-surface engineering."]
            action = "Request drone imagery or physical inspection."
            requires_drone = True

        result = ClaimVerificationResult(
            claim=activity,
            status=status.value,
            observations=obs,
            reason=reason,
            limitations=limits,
            recommended_next_action=action,
        )
        verification_results.append(result.model_dump())

    # Determine overall status
    if any(r["status"] == VerificationStatus.NOT_SUPPORTED.value for r in verification_results):
        overall_status = "POTENTIAL_CONTRADICTION"
    elif requires_drone or any(r["status"] == VerificationStatus.INSUFFICIENT_EVIDENCE.value for r in verification_results):
        overall_status = "INSUFFICIENT_EVIDENCE_REQUIRES_INSPECTION"
    else:
        overall_status = "SATELLITE_EVIDENCE_SUPPORTED"

    # Drone inspection request generation
    drone_request = None
    if requires_drone:
        unresolved_claims = [
            r["claim"] for r in verification_results if r["status"] != VerificationStatus.SUPPORTED.value
        ]
        drone_request = {
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "unresolved_claims": unresolved_claims,
            "required_inspection_type": "High-Resolution Drone Orthophoto & Physical Site Inspection",
            "guidelines": (
                "Upload clear close-up photos of structural foundations, culverts, compaction layers, "
                "or material delivery receipts to verify non-visible engineering components."
            ),
        }

    sat_metrics_obj = None
    if metrics:
        try:
            sat_metrics_obj = SatelliteMetrics(**metrics)
        except Exception:
            sat_metrics_obj = None

    # Build Verification Report
    report = VerificationReport(
        case_id=case_id,
        project_name=project_name,
        construction_type=construction_type,
        location=location,
        start_date=start_date,
        completion_date=completion_date,
        overall_status=overall_status,
        claims_summary=[ClaimVerificationResult(**r) for r in verification_results],
        satellite_metrics=sat_metrics_obj,
        requires_drone_verification=requires_drone,
        drone_inspection_request=drone_request,
        audit_log=[
            f"Case {case_id} initialized at {datetime.now(timezone.utc).isoformat()}",
            f"Satellite spectral analysis completed (Valid coverage: {metrics.get('valid_observation_coverage_pct', 0)}%)",
            f"Synthesized {len(verification_results)} claim verification assessments",
        ],
    )

    report_dict = report.model_dump()

    # Formulate officer summary message
    lines = [
        f"### Construction Claim Verification Report — Case `{case_id}`",
        f"**Project**: {project_name} | **Type**: {construction_type}",
        f"**Status**: `{overall_status}`",
        "",
        "| Claim | Status | Action Recommended |",
        "| :--- | :--- | :--- |",
    ]
    for r in verification_results:
        st_icon = "🟢" if r["status"] == "SUPPORTED" else "🔴" if r["status"] == "NOT_SUPPORTED" else "🟡"
        lines.append(f"| **{r['claim']}** | {st_icon} {r['status']} | {r['recommended_next_action']} |")

    lines.append("")
    if requires_drone:
        lines.append("> [!IMPORTANT]")
        lines.append("> **Inspection Required**: Satellite evidence alone is insufficient for structural/sub-surface claims. Please upload high-resolution drone or on-site inspection photographs below.")
    else:
        lines.append("> [!NOTE]")
        lines.append("> Visible claims are supported by multi-temporal satellite evidence. Payment decision awaits explicit human approval.")

    lines.append("")
    lines.append("**Officer Action Needed**: Review the evidence above and select **Approve Payment**, **Reject Claim**, or **Request Further Verification**.")

    return {
        "verification_results": verification_results,
        "overall_status": overall_status,
        "requires_drone_verification": requires_drone,
        "report": report_dict,
        "messages": [AIMessage(content="\n".join(lines))],
    }
