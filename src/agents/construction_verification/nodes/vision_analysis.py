import logging
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.construction_verification.services.gemini_client import GeminiVerificationClient
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)


async def vision_analysis_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict[str, Any]:
    """Execute Gemini Multimodal Vision analysis on pre/post satellite images and claims."""
    metrics = state.get("satellite_metrics") or {}
    claims = state.get("construction_claims") or []
    before_scene = state.get("before_scene") or {}
    after_scene = state.get("after_scene") or {}
    errors = list(state.get("errors", []))

    rgb_before = metrics.get("rgb_before_preview_path")
    rgb_after = metrics.get("rgb_after_preview_path")
    before_date = before_scene.get("acquisition_date", state.get("start_date", "Pre-construction"))
    after_date = after_scene.get("acquisition_date", state.get("completion_date", "Post-construction"))

    if not rgb_before or not rgb_after or not claims:
        err = "Missing imagery preview artifacts or claims for vision analysis."
        logger.warning(err)
        return {
            "errors": errors + [err],
            "vision_findings": {"findings": []},
            "vision_audit": {
                "vision_actually_executed": False,
                "api_call_status": "NOT_ATTEMPTED",
                "parsed_response_status": "VISION_NOT_EXECUTED",
                "error_message": err,
            },
            "messages": [AIMessage(content=f"Skipping vision analysis: {err}")],
        }

    gemini_client = GeminiVerificationClient()

    try:
        vision_output = await gemini_client.analyze_satellite_imagery(
            claims=claims,
            before_image_path=rgb_before,
            after_image_path=rgb_after,
            metrics=metrics,
            before_date=before_date,
            after_date=after_date,
            before_scene_id=before_scene.get("scene_id"),
            after_scene_id=after_scene.get("scene_id"),
            raster_dimensions=tuple(metrics["raster_shape"]) if metrics.get("raster_shape") else None,
        )

        findings_dicts = [f.model_dump() for f in vision_output.findings]

        msg_lines = ["**Gemini Multimodal Satellite Vision Findings:**"]
        for f in findings_dicts:
            msg_lines.append(f"- **{f.get('claim')}**: {f.get('status')}\n  _{f.get('reasoning')}_")

        audit_dict = vision_output.vision_audit.model_dump() if getattr(vision_output, "vision_audit", None) else {}
        return {
            "vision_findings": {
                "findings": findings_dicts,
                "general_observations": vision_output.general_observations,
                "visible_progress": vision_output.visible_progress,
                "progress_statement": vision_output.progress_statement,
            },
            "vision_audit": audit_dict,
            "messages": [AIMessage(content="\n".join(msg_lines))],
        }

    except Exception as e:
        err = f"Vision analysis error: {e}"
        logger.error(err)
        return {
            "errors": errors + [err],
            "vision_findings": {"findings": []},
            "vision_audit": {
                "vision_actually_executed": False,
                "api_call_status": "FAILED",
                "parsed_response_status": "VISION_NOT_EXECUTED",
                "error_message": str(e),
            },
            "messages": [AIMessage(content=err)],
        }

