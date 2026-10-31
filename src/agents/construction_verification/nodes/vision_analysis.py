import logging
from pathlib import Path
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
    metrics = state.get("satellite_metrics", {})
    claims = state.get("construction_claims", [])
    before_scene = state.get("before_scene", {})
    after_scene = state.get("after_scene", {})
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
        )

        findings_dicts = [f.model_dump() for f in vision_output.findings]

        msg_lines = ["**Gemini Multimodal Satellite Vision Findings:**"]
        for f in findings_dicts:
            status_indicator = (
                "✅ Visible changes support claim"
                if f.get("supports_claim") is True
                else "❌ Contradicted by imagery"
                if f.get("supports_claim") is False
                else "⚠️ Inconclusive / Insufficient satellite resolution"
            )
            msg_lines.append(f"- **{f.get('claim')}**: {status_indicator}\n  _{f.get('reasoning')}_")

        return {
            "vision_findings": {
                "findings": findings_dicts,
                "general_observations": vision_output.general_observations,
            },
            "messages": [AIMessage(content="\n".join(msg_lines))],
        }

    except Exception as e:
        err = f"Gemini Vision analysis error: {e}"
        logger.error(err)
        return {
            "errors": errors + [err],
            "messages": [AIMessage(content=err)],
        }

