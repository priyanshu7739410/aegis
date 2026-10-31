import logging
import uuid
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.construction_verification.services.raster_processing import RasterProcessingError, RasterProcessor
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)


async def satellite_analysis_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict[str, Any]:
    """Execute deterministic NDVI, BSI, and spectral change analysis."""
    case_id = state.get("case_id") or str(uuid.uuid4())[:8]
    bbox = state.get("bounding_box")
    before_scene = state.get("before_scene", {})
    after_scene = state.get("after_scene", {})
    errors = list(state.get("errors", []))

    if not before_scene or not after_scene or not bbox:
        err = "Missing satellite scenes or bounding box for change analysis."
        logger.error(err)
        return {
            "errors": errors + [err],
            "satellite_error": state.get("satellite_error") or err,
            "satellite_metrics": None,
            "messages": [AIMessage(content=f"Skipping satellite analysis: {err}")],
        }

    processor = RasterProcessor()

    try:
        metrics = processor.analyze_change(
            case_id=case_id,
            bbox_wgs84=bbox,
            before_scene=before_scene,
            after_scene=after_scene,
        )

        bsi_line = (
            f"- **Mean Bare Soil Index (BSI)**: Before {metrics['mean_bsi_before']:.3f} -> After {metrics['mean_bsi_after']:.3f} (Δ {metrics['delta_bsi']:+.3f})\n"
            if metrics.get("delta_bsi") is not None
            else "- **Bare Soil Index (BSI)**: not available (no SWIR band in the scene)\n"
        )
        msg = (
            f"Remote Sensing Change Analysis Completed:\n"
            f"- **Mean NDVI Before**: {metrics['mean_ndvi_before']:.3f} | **After**: {metrics['mean_ndvi_after']:.3f} (Δ {metrics['delta_ndvi']:+.3f})\n"
            f"- **Vegetation Loss Fraction**: {metrics['vegetation_loss_fraction'] * 100:.1f}% of site area\n"
            f"- **Estimated Disturbed Ground**: {metrics['potential_disturbed_area_m2']:,.0f} m²\n"
            + bsi_line
            + f"- **Valid Cloud-Free Observation**: {metrics['valid_observation_coverage_pct']:.1f}%\n"
        )

        return {
            "case_id": case_id,
            "satellite_metrics": metrics,
            "satellite_error": None,
            "messages": [AIMessage(content=msg)],
        }

    except RasterProcessingError as e:
        err = f"Raster analysis error: {e}"
        logger.error(err)
        return {
            "errors": errors + [err],
            "satellite_error": str(e),
            "satellite_metrics": None,
            "messages": [AIMessage(content=f"⚠️ Satellite Analysis Failed: {e}")],
        }
    except Exception as e:
        err = f"Unexpected error during satellite analysis: {e}"
        logger.error(err)
        return {
            "errors": errors + [err],
            "satellite_error": str(e),
            "satellite_metrics": None,
            "messages": [AIMessage(content=f"⚠️ Satellite Analysis Error: {e}")],
        }

