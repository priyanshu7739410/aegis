from datetime import date
import logging
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.construction_verification.services.earth_search import EarthSearchClient, EarthSearchError
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)


def _input_problem(bbox: list[float] | None, start_date: str | None, completion_date: str | None) -> str | None:
    if not bbox or len(bbox) != 4:
        return "Valid bounding box coordinates are required for satellite imagery search."
    min_lon, min_lat, max_lon, max_lat = bbox
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        return f"Invalid coordinates: bounding box {bbox} is outside the valid WGS84 range."
    if not start_date or not completion_date:
        return "Both start date and completion date are required to retrieve before and after imagery."
    try:
        if date.fromisoformat(start_date[:10]) > date.fromisoformat(completion_date[:10]):
            return f"Start date {start_date} is after completion date {completion_date}."
    except ValueError:
        return f"Dates must be YYYY-MM-DD (got '{start_date}' and '{completion_date}')."
    return None


async def satellite_search_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict[str, Any]:
    """Discover pre-construction and post-construction Sentinel-2 scenes via Earth Search STAC."""
    bbox = state.get("bounding_box")
    start_date = state.get("start_date")
    completion_date = state.get("completion_date")
    errors = list(state.get("errors", []))

    err = _input_problem(bbox, start_date, completion_date)
    if err:
        logger.error(err)
        return {
            "errors": errors + [err],
            "satellite_error": err,
            "before_scene": {},
            "after_scene": {},
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
                "satellite_error": err,
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
            "satellite_error": None,
            "messages": [AIMessage(content=msg)],
        }

    except EarthSearchError as e:
        err = f"Earth Search STAC error: {e}"
        logger.error(err)
        return {
            "errors": errors + [err],
            "satellite_error": err,
            "before_scene": {},
            "after_scene": {},
            "messages": [AIMessage(content=f"Error discovering satellite imagery: {e}")],
        }
    except Exception as e:
        err = f"Unexpected error during satellite scene search: {e}"
        logger.error(err)
        return {
            "errors": errors + [err],
            "satellite_error": err,
            "before_scene": {},
            "after_scene": {},
            "messages": [AIMessage(content=err)],
        }

