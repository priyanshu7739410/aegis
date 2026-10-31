from datetime import datetime, timezone
import logging
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from core.settings import settings
from agents.construction_verification.schemas import (
    ClaimVerificationResult,
    SatelliteMetrics,
    VerificationReport,
    VerificationStatus,
)
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)

LOW_COVERAGE_PCT = 50.0
RESOLUTION_LIMITATION = (
    "Sentinel-2 resolves about 10 m per pixel. It cannot confirm sub-surface work, structural "
    "quality, material thickness, or features narrower than a few pixels."
)


def _is_synthetic(metrics: dict[str, Any], *scenes: dict[str, Any]) -> bool:
    return bool(metrics.get("synthetic")) or any(
        scene.get("properties", {}).get("synthetic") for scene in scenes
    )


def _matches(finding: dict[str, Any], activity: str) -> bool:
    claim = str(finding.get("claim") or "").strip().lower()
    target = activity.strip().lower()
    # An empty claim is a substring of everything, so it must never match.
    return bool(claim) and (claim == target or claim in target or target in claim)


def _evidence_limitations(
    state: ConstructionVerificationState,
    metrics: dict[str, Any],
    vision_executed: bool,
    synthetic: bool,
) -> list[str]:
    limitations = [RESOLUTION_LIMITATION]
    if synthetic:
        limitations.insert(0, "SYNTHETIC TEST DATA: this report contains no genuine satellite evidence.")
    if state.get("satellite_error"):
        limitations.append(f"Satellite evidence unavailable: {state['satellite_error']}")
    if not metrics:
        limitations.append("No spectral measurements were produced for this case.")
    else:
        coverage = metrics.get("valid_observation_coverage_pct", 0.0)
        if coverage < LOW_COVERAGE_PCT:
            limitations.append(
                f"Only {coverage}% of site pixels were cloud-free in both scenes; spectral values may not represent the whole site."
            )
        if metrics.get("delta_bsi") is None:
            limitations.append("Bare Soil Index was not computed because a SWIR band was unavailable.")
    if not vision_executed:
        audit_error = (state.get("vision_audit") or {}).get("error_message")
        limitations.append(f"The vision model did not assess the imagery. {audit_error or ''}".strip())
    return limitations


async def verification_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict[str, Any]:
    """Synthesize satellite metrics and vision findings into claim-by-claim verification report."""
    claims = state.get("construction_claims") or []
    vision_findings = (state.get("vision_findings") or {}).get("findings") or []
    metrics = state.get("satellite_metrics") or {}
    case_id = state.get("case_id") or "UNASSIGNED"
    project_name = state.get("project_name") or "Not provided"
    construction_type = state.get("construction_type") or "Not provided"
    before_scene = state.get("before_scene") or {}
    after_scene = state.get("after_scene") or {}
    location = state.get("construction_location") or {}
    start_date = state.get("start_date")
    completion_date = state.get("completion_date")

    verification_results: list[dict[str, Any]] = []
    requires_drone = False

    vision_audit = state.get("vision_audit") or {}
    vision_executed = bool(vision_audit.get("vision_actually_executed", False))

    synthetic = settings.USE_FAKE_MODEL or _is_synthetic(metrics, before_scene, after_scene)
    if synthetic and not settings.USE_FAKE_MODEL:
        logger.error("Discarding synthetic evidence found outside test mode.")
        metrics, before_scene, after_scene, vision_findings = {}, {}, {}, []
        vision_executed = False
        synthetic = False

    # Match each claim with vision finding & deterministic metrics
    for claim_dict in claims:
        activity = claim_dict.get("activity", "")
        # Find matching vision finding
        finding = next((f for f in vision_findings if _matches(f, activity)), None)

        if not vision_executed:
            status = VerificationStatus.VISION_NOT_EXECUTED
            obs = []
            reason = (finding or {}).get("reasoning") or (
                "VISION_NOT_EXECUTED: The vision model did not assess this claim."
            )
            limits = (finding or {}).get("limitations") or ["Image analysis did not run."]
            action = "Vision analysis not executed. Re-run the analysis or arrange a physical inspection."
            requires_drone = True
        elif finding:
            status = VerificationStatus(finding.get("status") or "INSUFFICIENT_EVIDENCE")
            obs = finding.get("observed_changes", [])
            reason = finding.get("reasoning", "")
            limits = finding.get("limitations", [])
            default_action = {
                VerificationStatus.SUPPORTED: "Visible evidence supports this claim. Officer review is still required.",
                VerificationStatus.NOT_SUPPORTED: "Ask the contractor to explain the contradiction and arrange an inspection.",
            }.get(status, "Satellite imagery is inconclusive. Arrange an on-site inspection.")
            action = finding.get("recommended_follow_up") or default_action
            if status != VerificationStatus.SUPPORTED:
                requires_drone = True
        else:
            status = VerificationStatus.INSUFFICIENT_EVIDENCE
            obs = []
            reason = "The vision model returned no finding for this claim."
            limits = [RESOLUTION_LIMITATION]
            action = "Arrange an on-site inspection."
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
    if not verification_results:
        overall_status = "NO_CLAIMS_ASSESSED"
        requires_drone = True
    elif any(r["status"] == VerificationStatus.NOT_SUPPORTED.value for r in verification_results):
        overall_status = "POTENTIAL_CONTRADICTION"
    elif any(r["status"] == VerificationStatus.INSUFFICIENT_EVIDENCE.value for r in verification_results):
        overall_status = "INSUFFICIENT_EVIDENCE_REQUIRES_INSPECTION"
    elif any(r["status"] == VerificationStatus.VISION_NOT_EXECUTED.value for r in verification_results):
        overall_status = "VISION_NOT_EXECUTED_REQUIRES_INSPECTION"
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
            "required_inspection_type": "Physical Site Inspection",
            "guidelines": (
                "Upload clear close-up photos of structural foundations, culverts, compaction layers, "
                "or material delivery receipts to verify non-visible engineering components."
            ),
        }

    sat_metrics_obj = SatelliteMetrics(**metrics) if metrics else None
    general_observations = (state.get("vision_findings") or {}).get("general_observations")

    audit_log = [f"Case {case_id} report generated at {datetime.now(timezone.utc).isoformat()}"]
    if metrics:
        audit_log.append(
            f"Satellite spectral analysis completed (valid coverage: {metrics.get('valid_observation_coverage_pct')}%)"
        )
    else:
        audit_log.append("Satellite spectral analysis did not produce measurements")
    audit_log.append(f"Vision executed: {vision_executed}")
    audit_log.append(f"Synthesized {len(verification_results)} claim verification assessments")

    # Build Verification Report
    report = VerificationReport(
        case_id=case_id,
        project_name=project_name,
        construction_type=construction_type,
        contractor=state.get("contractor"),
        payment_amount=state.get("payment_amount"),
        evidence_mode="SYNTHETIC_TEST_DATA" if synthetic else "LIVE_SERVICES",
        before_scene=before_scene or None,
        after_scene=after_scene or None,
        vision_audit=vision_audit or None,
        general_observations=general_observations if vision_executed else None,
        evidence_limitations=_evidence_limitations(state, metrics, vision_executed, synthetic),
        location=location,
        start_date=start_date,
        completion_date=completion_date,
        overall_status=overall_status,
        claims_summary=[ClaimVerificationResult(**r) for r in verification_results],
        satellite_metrics=sat_metrics_obj,
        requires_drone_verification=requires_drone,
        drone_inspection_request=drone_request,
        audit_log=audit_log,
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
        lines.append("> **Inspection Recommended**: Satellite evidence alone did not establish every claim.")
    else:
        lines.append("> [!NOTE]")
        lines.append("> Visible claims are supported by multi-temporal satellite evidence. Payment decision awaits explicit human approval.")

    lines.append("")
    lines.append("**Officer Action Needed**: This report does not authorize payment. A human officer must review the evidence and record a decision.")

    return {
        "verification_results": verification_results,
        "overall_status": overall_status,
        "requires_drone_verification": requires_drone,
        "report": report_dict,
        "messages": [AIMessage(content="\n".join(lines))],
    }
