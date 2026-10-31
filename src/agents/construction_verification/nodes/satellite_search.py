import logging
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.construction_verification.services.earth_search import EarthSearchClient, EarthSearchError
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)


async def satellite_search_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict[str, Any]:
    """Discover pre-construction and post-construction Sentinel-2 scenes via Earth Search STAC."""
    bbox = state.get("bounding_box")
    start_date = state.get("start_date")
    completion_date = state.get("completion_date")
    errors = list(state.get("errors", []))

    if not bbox or len(bbox) != 4:
        err = "Valid bounding box coordinates are required for satellite imagery search."
        logger.error(err)
        return {
            "errors": errors + [err],
            "messages": [AIMessage(content=f"Cannot search satellite scenes: {err}")],
        }

    if not start_date or not completion_date:
        err = "Both start date and completion date are required to retrieve before and after imagery."
        logger.error(err)
        return {
            "errors": errors + [err],
            "messages": [AIMessage(content=f"Cannot search satellite scenes: {err}")],
        }

    client = EarthSearchClient()

    try:
        # Search before scene: target around project start date (preferring scenes prior to or at start)
        logger.info(f"Searching pre-construction Sentinel-2 scene around {start_date} for bbox {bbox}...")
        before_scene = client.find_best_scene(
            bbox=bbox, target_date=start_date, days_window=60, prefer_before=True
        )

        # Search after scene: target around completion date (scenes after completion date)
        logger.info(f"Searching post-construction Sentinel-2 scene around {completion_date} for bbox {bbox}...")
        after_scene = client.find_best_scene(
            bbox=bbox, target_date=completion_date, days_window=60, prefer_before=False
        )

        if not before_scene or not after_scene:
            err = (
                f"Satellite imagery search incomplete. Found before_scene: {bool(before_scene)}, "
                f"after_scene: {bool(after_scene)}. Cloud cover or spatial coverage may be insufficient."
            )
            logger.warning(err)
            return {
                "errors": errors + [err],
                "before_scene": before_scene or {},
                "after_scene": after_scene or {},
                "messages": [AIMessage(content=f"Warning: {err}")],
            }

        msg = (
            f"Retrieved Sentinel-2 Satellite Scenes:\n"
            f"- **Before Scene**: `{before_scene['scene_id']}` (Acquired: {before_scene['acquisition_date']}, Cloud: {before_scene['cloud_cover_percent']}%\n"
            f"- **After Scene**: `{after_scene['scene_id']}` (Acquired: {after_scene['acquisition_date']}, Cloud: {after_scene['cloud_cover_percent']}%\n"
        )

        return {
            "before_scene": before_scene,
            "after_scene": after_scene,
            "messages": [AIMessage(content=msg)],
        }

    except EarthSearchError as e:
        err = f"Earth Search STAC error: {e}"
        logger.error(err)
        return {
            "errors": errors + [err],
            "messages": [AIMessage(content=f"Error discovering satellite imagery: {e}")],
        }
    except Exception as e:
        err = f"Unexpected error during satellite scene search: {e}"
        logger.error(err)
        return {
            "errors": errors + [err],
            "messages": [AIMessage(content=err)],
        }

