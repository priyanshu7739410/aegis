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

        msg = (
            f"Remote Sensing Change Analysis Completed:\n"
            f"- **Mean NDVI Before**: {metrics.get('mean_ndvi_before', 0.0):.3f} | **After**: {metrics.get('mean_ndvi_after', 0.0):.3f} (Δ {metrics.get('delta_ndvi', 0.0):+.3f})\n"
            f"- **Vegetation Loss Fraction**: {metrics.get('vegetation_loss_fraction', 0.0) * 100:.1f}% of site area\n"
            f"- **Estimated Disturbed Ground**: {metrics.get('potential_disturbed_area_m2', 0.0):,.0f} m²\n"
            f"- **Mean Bare Soil Index (BSI)**: Before {metrics.get('mean_bsi_before', 0.0):.3f} -> After {metrics.get('mean_bsi_after', 0.0):.3f} (Δ {metrics.get('delta_bsi', 0.0):+.3f})\n"
            f"- **Valid Cloud-Free Observation**: {metrics.get('valid_observation_coverage_pct', 0.0):.1f}%\n"
        )

        return {
            "case_id": case_id,
            "satellite_metrics": metrics,
            "messages": [AIMessage(content=msg)],
        }

    except RasterProcessingError as e:
        err = f"Raster analysis error: {e}"
        logger.error(err)
        return {
            "errors": errors + [err],
            "messages": [AIMessage(content=err)],
        }
    except Exception as e:
        err = f"Unexpected error during satellite analysis: {e}"
        logger.error(err)
        return {
            "errors": errors + [err],
            "messages": [AIMessage(content=err)],
        }

