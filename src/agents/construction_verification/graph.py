import logging
from typing import Literal
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agents.construction_verification.nodes.drone_eval import drone_eval_node
from agents.construction_verification.nodes.extract_claims import extract_claims_node
from agents.construction_verification.nodes.ocr import ocr_node
from agents.construction_verification.nodes.satellite_analysis import satellite_analysis_node
from agents.construction_verification.nodes.satellite_search import satellite_search_node
from agents.construction_verification.nodes.verification import verification_node
from agents.construction_verification.nodes.vision_analysis import vision_analysis_node
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)


import logging
import re
from typing import Any, Literal
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agents.construction_verification.nodes.drone_eval import drone_eval_node
from agents.construction_verification.nodes.extract_claims import extract_claims_node
from agents.construction_verification.nodes.ocr import ocr_node
from agents.construction_verification.nodes.satellite_analysis import satellite_analysis_node
from agents.construction_verification.nodes.satellite_search import satellite_search_node
from agents.construction_verification.nodes.verification import verification_node
from agents.construction_verification.nodes.vision_analysis import vision_analysis_node
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)


def route_entry(
    state: ConstructionVerificationState,
) -> Literal["drone_eval", "ocr", "extract_claims", "satellite_search", "chat_handler"]:
    """Determine initial node based on incoming state data and user intent."""
    messages = state.get("messages", [])
    has_human_msg = False
    last_text = ""
    if messages and isinstance(messages[-1], HumanMessage):
        has_human_msg = True
        last_text = str(messages[-1].content or "").strip()

    # If incoming input is a chat message from officer
    if has_human_msg and last_text:
        # Check if user wants to run or verify
        is_verify = bool(re.search(r"\b(verify|check|run|start|evaluate)\b", last_text, re.IGNORECASE))
        has_location_dates = bool(
            state.get("bounding_box") and state.get("start_date") and state.get("completion_date")
        )

        if is_verify and has_location_dates:
            return "satellite_search"

        # If user explicitly requested verification but claims/location are not yet extracted, prioritize extraction
        if is_verify and not state.get("bounding_box") and not state.get("construction_claims"):
            if state.get("extracted_text"):
                return "extract_claims"
            if state.get("pdf_path") or state.get("image_path"):
                return "ocr"

        # Otherwise handle through chat handler (extracts inline coords/dates or answers inquiry)
        return "chat_handler"

    # Direct workflow execution path
    if state.get("drone_image_paths") and state.get("verification_results"):
        return "drone_eval"

    if state.get("bounding_box") and state.get("start_date") and state.get("completion_date") and state.get("construction_claims"):
        return "satellite_search"

    if state.get("extracted_text"):
        return "extract_claims"

    return "ocr"


async def chat_handler_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict[str, Any]:
    """Process officer conversational chat messages, extract inline coordinates/dates, and provide case answers."""
    messages = state.get("messages", [])
    last_msg = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
    text = str(last_msg).strip()

    updates: dict[str, Any] = {}
    manual_corrections = dict(state.get("manual_corrections", {}))

    # 1. Parse coordinates if supplied in chat
    lat = None
    lon = None
    lat_match = re.search(r"(?:latitude|lat)[:\s=]+([-+]?\d{1,2}(?:\.\d+)?)", text, re.IGNORECASE)
    lon_match = re.search(r"(?:longitude|lon)[:\s=]+([-+]?\d{1,3}(?:\.\d+)?)", text, re.IGNORECASE)

    if lat_match and lon_match:
        try:
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))
        except ValueError:
            pass
    else:
        # Check general coordinate pair: 22.7215, 88.4812
        pair = re.search(r"\b([-+]?\d{1,2}\.\d{3,})[,\s]+([-+]?\d{1,3}\.\d{3,})\b", text)
        if pair:
            try:
                lat = float(pair.group(1))
                lon = float(pair.group(2))
            except ValueError:
                pass

    if lat is not None and lon is not None and (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0):
        bbox = [
            round(lon - 0.005, 6),
            round(lat - 0.005, 6),
            round(lon + 0.005, 6),
            round(lat + 0.005, 6),
        ]
        location_dict = {
            "site_name": state.get("project_name", "Case Project Site"),
            "latitude": lat,
            "longitude": lon,
            "bbox": bbox,
            "preliminary_search_location": True,
        }
        updates["construction_location"] = location_dict
        updates["bounding_box"] = bbox
        manual_corrections["latitude"] = {"value": lat, "provenance": "officer-chat"}
        manual_corrections["longitude"] = {"value": lon, "provenance": "officer-chat"}

    # 2. Parse dates if supplied in chat
    date_matches = re.findall(r"\b(20\d\d[-/]\d\d[-/]\d\d)\b", text)
    if len(date_matches) >= 2:
        norm_start = date_matches[0].replace("/", "-")
        norm_comp = date_matches[1].replace("/", "-")
        updates["start_date"] = norm_start
        updates["completion_date"] = norm_comp
        manual_corrections["start_date"] = {"value": norm_start, "provenance": "officer-chat"}
        manual_corrections["completion_date"] = {"value": norm_comp, "provenance": "officer-chat"}
    elif len(date_matches) == 1:
        if "start" in text.lower():
            norm_start = date_matches[0].replace("/", "-")
            updates["start_date"] = norm_start
            manual_corrections["start_date"] = {"value": norm_start, "provenance": "officer-chat"}
        elif "comp" in text.lower() or "end" in text.lower():
            norm_comp = date_matches[0].replace("/", "-")
            updates["completion_date"] = norm_comp
            manual_corrections["completion_date"] = {"value": norm_comp, "provenance": "officer-chat"}

    if manual_corrections:
        updates["manual_corrections"] = manual_corrections

    # Check effective state
    eff_lat = updates.get("construction_location", {}).get("latitude", state.get("construction_location", {}).get("latitude"))
    eff_lon = updates.get("construction_location", {}).get("longitude", state.get("construction_location", {}).get("longitude"))
    eff_start = updates.get("start_date", state.get("start_date"))
    eff_comp = updates.get("completion_date", state.get("completion_date"))
    eff_claims = state.get("construction_claims", [])
    project = state.get("project_name", "Construction Project")

    # Formulate conversational reply
    reply_lines = []
    if lat is not None and lon is not None:
        reply_lines.append(f"📍 **Coordinates Updated**: Centroid set to `({lat}, {lon})` (preliminary search bounding box computed).")
    if "start_date" in updates or "completion_date" in updates:
        reply_lines.append(f"📅 **Dates Updated**: Construction period set to `{eff_start or 'N/A'}` to `{eff_comp or 'N/A'}`.")

    if eff_lat is not None and eff_lon is not None and eff_start and eff_comp:
        reply_lines.append(
            "\n✅ **All required parameters are valid.**\n"
            f"- Project: **{project}**\n"
            f"- Location: `({eff_lat}, {eff_lon})`\n"
            f"- Timeline: `{eff_start}` to `{eff_comp}`\n"
            f"- Claims: {len(eff_claims)} item(s)\n\n"
            "You can type **'verify'** or click **'🛰️ Run Satellite Verification'** in Step 3 to proceed."
        )
    else:
        missing = []
        if eff_lat is None or eff_lon is None:
            missing.append("Site geographic coordinates (latitude, longitude)")
        if not eff_start:
            missing.append("Project start date (YYYY-MM-DD)")
        if not eff_comp:
            missing.append("Claimed completion date (YYYY-MM-DD)")
        if not state.get("extracted_text"):
            missing.append("Construction document (PDF or text upload)")

        reply_lines.append(
            f"ℹ️ **Current Case Status for {state.get('case_id', 'Case')}**:\n"
            f"- Project: **{project}**\n"
            + "\n".join([f"- ⚠️ Missing: {m}" for m in missing])
            + "\n\nPlease supply the missing values in the dashboard tabs or directly in chat."
        )

    updates["messages"] = [AIMessage(content="\n\n".join(reply_lines))]
    return updates


def route_after_validation(
    state: ConstructionVerificationState,
) -> Literal["satellite_search", "clarification"]:
    """Check if location and dates are present before querying satellite STAC."""
    bbox = state.get("bounding_box")
    start_date = state.get("start_date")
    completion_date = state.get("completion_date")

    if bbox and start_date and completion_date:
        return "satellite_search"
    return "clarification"


async def clarification_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict:
    """Prompt the officer to provide missing coordinates, dates, or claims."""
    missing: list[str] = []
    if not state.get("bounding_box"):
        missing.append("Site geographic coordinates / bounding box (latitude and longitude)")
    if not state.get("start_date"):
        missing.append("Project start date (YYYY-MM-DD)")
    if not state.get("completion_date"):
        missing.append("Claimed completion date (YYYY-MM-DD)")

    msg = (
        "### ⚠️ Clarification Required Before Satellite Verification\n\n"
        "The submitted document did not contain sufficient geographic or temporal information to locate Sentinel-2 satellite imagery.\n\n"
        f"**Missing Required Information:**\n"
        + "\n".join([f"- {m}" for m in missing])
        + "\n\nPlease provide the site coordinates (latitude, longitude) and construction dates in the fields above or via message to proceed."
    )
    return {"messages": [AIMessage(content=msg)]}


# Assemble the LangGraph workflow
workflow = StateGraph(ConstructionVerificationState)

workflow.add_node("ocr", ocr_node)
workflow.add_node("extract_claims", extract_claims_node)
workflow.add_node("clarification", clarification_node)
workflow.add_node("chat_handler", chat_handler_node)
workflow.add_node("satellite_search", satellite_search_node)
workflow.add_node("satellite_analysis", satellite_analysis_node)
workflow.add_node("vision_analysis", vision_analysis_node)
workflow.add_node("verification", verification_node)
workflow.add_node("drone_eval", drone_eval_node)

# Flow edges
workflow.set_conditional_entry_point(
    route_entry,
    {
        "drone_eval": "drone_eval",
        "ocr": "ocr",
        "extract_claims": "extract_claims",
        "satellite_search": "satellite_search",
        "chat_handler": "chat_handler",
    },
)

workflow.add_edge("chat_handler", END)
workflow.add_edge("ocr", "extract_claims")
workflow.add_conditional_edges(
    "extract_claims",
    route_after_validation,
    {
        "satellite_search": "satellite_search",
        "clarification": "clarification",
    },
)
workflow.add_edge("clarification", END)

workflow.add_edge("satellite_search", "satellite_analysis")
workflow.add_edge("satellite_analysis", "vision_analysis")
workflow.add_edge("vision_analysis", "verification")
workflow.add_edge("verification", END)
workflow.add_edge("drone_eval", END)

# Compile agent graph
construction_verification_agent = workflow.compile()

