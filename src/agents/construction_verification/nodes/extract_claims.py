import logging
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.construction_verification.services.gemini_client import GeminiVerificationClient
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)


async def extract_claims_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict[str, Any]:
    """Parse OCR text using Gemini into structured project metadata and claims."""
    extracted_text = state.get("extracted_text", "").strip()
    errors = list(state.get("errors", []))

    if not extracted_text:
        err = "Cannot extract claims because document text is empty."
        return {
            "errors": errors + [err],
            "messages": [AIMessage(content=err)],
        }

    gemini_client = GeminiVerificationClient()
    info = await gemini_client.extract_project_info(extracted_text)

    # Resolve coordinates and bounding box with manual officer corrections taking precedence
    loc = info.location
    lat = loc.latitude if loc else None
    lon = loc.longitude if loc else None
    bbox = loc.bbox if loc else None
    start_date = info.start_date
    completion_date = info.completion_date

    manual = state.get("manual_corrections", {})
    if "latitude" in manual:
        val = manual["latitude"].get("value", manual["latitude"]) if isinstance(manual["latitude"], dict) else manual["latitude"]
        if val is not None:
            lat = float(val)
    elif state.get("construction_location", {}).get("latitude") is not None:
        lat = float(state["construction_location"]["latitude"])

    if "longitude" in manual:
        val = manual["longitude"].get("value", manual["longitude"]) if isinstance(manual["longitude"], dict) else manual["longitude"]
        if val is not None:
            lon = float(val)
    elif state.get("construction_location", {}).get("longitude") is not None:
        lon = float(state["construction_location"]["longitude"])

    if "start_date" in manual:
        val = manual["start_date"].get("value", manual["start_date"]) if isinstance(manual["start_date"], dict) else manual["start_date"]
        if val:
            start_date = str(val).strip().replace("/", "-")
    elif state.get("start_date"):
        start_date = str(state["start_date"]).strip().replace("/", "-")

    if "completion_date" in manual:
        val = manual["completion_date"].get("value", manual["completion_date"]) if isinstance(manual["completion_date"], dict) else manual["completion_date"]
        if val:
            completion_date = str(val).strip().replace("/", "-")
    elif state.get("completion_date"):
        completion_date = str(state["completion_date"]).strip().replace("/", "-")

    if lat is not None and lon is not None:
        bbox = [
            round(lon - 0.005, 6),
            round(lat - 0.005, 6),
            round(lon + 0.005, 6),
            round(lat + 0.005, 6),
        ]
    elif state.get("bounding_box"):
        bbox = state["bounding_box"]

    claims_dicts = [c.model_dump() for c in info.claims]
    if "construction_claims" in state and state["construction_claims"]:
        claims_dicts = state["construction_claims"]

    missing_fields = list(info.missing_fields)
    if lat is not None and lon is not None:
        missing_fields = [
            f for f in missing_fields
            if "coordinate" not in f.lower() and "location" not in f.lower() and f != "site_coordinates"
        ]
    if start_date and completion_date:
        missing_fields = [
            f for f in missing_fields
            if "date" not in f.lower() and f != "project_dates"
        ]

    location_dict = {
        "site_name": loc.site_name if loc else (state.get("project_name") or info.project_name),
        "latitude": lat,
        "longitude": lon,
        "bbox": bbox,
        "preliminary_search_location": True if lat is not None and lon is not None else False,
    }

    project_name = state.get("project_name") or info.project_name
    construction_type = state.get("construction_type") or info.construction_type

    status_message = (
        f"Extracted Project: **{project_name}** ({construction_type})\n"
        f"- Coordinates: {lat}, {lon}\n"
        f"- Date Range: {start_date} to {completion_date}\n"
        f"- Claims ({len(claims_dicts)}): {', '.join([c['activity'] for c in claims_dicts])}\n"
    )
    if missing_fields:
        status_message += f"\n> [!WARNING] Missing Required Information: {', '.join(missing_fields)}"

    # Clean previous missing-fields errors if now resolved
    cleaned_errors = [e for e in errors if not e.startswith("Missing required fields")]
    if missing_fields:
        cleaned_errors.append(f"Missing required fields: {', '.join(missing_fields)}")

    return {
        "project_name": project_name,
        "construction_type": construction_type,
        "construction_location": location_dict,
        "construction_claims": claims_dicts,
        "start_date": start_date or "",
        "completion_date": completion_date or "",
        "bounding_box": bbox,
        "errors": cleaned_errors,
        "messages": [AIMessage(content=status_message)],
    }

