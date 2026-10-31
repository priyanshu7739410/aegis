import logging
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict
from langgraph.graph import END, START, StateGraph

from agents.construction_verification.schemas import DiscoveryReport
from agents.construction_verification.services.discovery_service import (
    ConstructionDiscoveryService,
    DiscoveryError,
)

logger = logging.getLogger(__name__)


class DiscoveryWorkflowState(TypedDict, total=False):
    """Independent Construction Discovery LangGraph State."""
    discovery_id: str
    latitude: float
    longitude: float
    radius_km: float
    before_date_start: str
    before_date_end: str
    after_date_start: str
    after_date_end: str
    min_change_area_m2: float
    max_cloud_cover: float
    area_name: str

    bounding_box: list[float]
    before_scene: dict[str, Any]
    after_scene: dict[str, Any]
    aligned_data: dict[str, Any]
    detection_result: dict[str, Any]
    candidates: list[dict[str, Any]]
    vision_result: dict[str, Any]
    report: dict[str, Any]
    audit: dict[str, Any]
    errors: list[str]


def validate_area_node(state: DiscoveryWorkflowState) -> dict[str, Any]:
    """Validate geographic coordinates and derive search bounding box."""
    lat = state.get("latitude", 28.17)
    lon = state.get("longitude", 77.61)
    radius_km = state.get("radius_km", 5.0)

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return {"errors": [f"Invalid coordinate centroid: lat={lat}, lon={lon}"]}

    bbox = ConstructionDiscoveryService.calculate_bounding_box(lat, lon, radius_km)
    disc_id = state.get("discovery_id") or f"DISC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    return {
        "discovery_id": disc_id,
        "bounding_box": bbox,
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius_km,
        "before_date_start": state.get("before_date_start", "2021-01-01"),
        "before_date_end": state.get("before_date_end", "2021-03-31"),
        "after_date_start": state.get("after_date_start", "2025-01-01"),
        "after_date_end": state.get("after_date_end", "2025-03-31"),
        "min_change_area_m2": state.get("min_change_area_m2", 5000.0),
        "max_cloud_cover": state.get("max_cloud_cover", 20.0),
        "area_name": state.get("area_name", "Independent Site Discovery"),
    }


def retrieve_before_after_imagery_node(state: DiscoveryWorkflowState) -> dict[str, Any]:
    """Query Earth Search STAC for low-cloud Sentinel-2 before and after scenes."""
    service = ConstructionDiscoveryService()
    bbox = state["bounding_box"]
    max_cloud = state.get("max_cloud_cover", 20.0)

    range_before = f"{state['before_date_start']}T00:00:00Z/{state['before_date_end']}T23:59:59Z"
    range_after = f"{state['after_date_start']}T00:00:00Z/{state['after_date_end']}T23:59:59Z"

    b_scene = service.earth_search.search_scene(bbox=bbox, datetime_range=range_before, max_cloud=max_cloud)
    a_scene = service.earth_search.search_scene(bbox=bbox, datetime_range=range_after, max_cloud=max_cloud)

    if not b_scene or not a_scene:
        err = f"Failed retrieving Sentinel-2 scenes: before_found={bool(b_scene)}, after_found={bool(a_scene)}"
        return {"errors": [err]}

    return {
        "before_scene": b_scene,
        "after_scene": a_scene,
    }


def preprocess_and_align_images_node(state: DiscoveryWorkflowState) -> dict[str, Any]:
    """Window-read multi-spectral bands and align to common CRS and pixel grid."""
    if state.get("errors"):
        return {}

    service = ConstructionDiscoveryService()
    try:
        aligned = service._read_and_align_rasters(
            bbox_wgs84=state["bounding_box"],
            before_scene=state["before_scene"],
            after_scene=state["after_scene"],
        )
        return {"aligned_data": aligned}
    except Exception as e:
        return {"errors": [f"Preprocess & alignment failed: {e}"]}


def detect_candidate_changes_node(state: DiscoveryWorkflowState) -> dict[str, Any]:
    """Perform NDVI/spectral change detection, SCL cloud masking, and vectorize candidate polygons."""
    if state.get("errors"):
        return {}

    service = ConstructionDiscoveryService()
    disc_id = state["discovery_id"]
    run_dir = service.output_dir / disc_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        detection = service._detect_changes(
            aligned_data=state["aligned_data"],
            bbox_wgs84=state["bounding_box"],
            min_change_area_m2=state.get("min_change_area_m2", 5000.0),
            run_dir=run_dir,
        )
        return {
            "detection_result": detection,
            "candidates": detection["candidates"],
        }
    except Exception as e:
        return {"errors": [f"Candidate change detection failed: {e}"]}


def analyze_candidates_with_gemini_vision_node(state: DiscoveryWorkflowState) -> dict[str, Any]:
    """Execute multimodal Gemini Vision analysis sending actual before/after image bytes."""
    if state.get("errors"):
        return {}

    service = ConstructionDiscoveryService()
    detection = state["detection_result"]
    before_scene = state["before_scene"]
    after_scene = state["after_scene"]

    scene_metadata = {
        "before_scene_id": before_scene["scene_id"],
        "after_scene_id": after_scene["scene_id"],
        "before_acquisition_date": before_scene["acquisition_date"],
        "after_acquisition_date": after_scene["acquisition_date"],
        "crs": state["aligned_data"]["crs"],
        "raster_shape": list(state["aligned_data"]["shape"]),
        "area_name": state.get("area_name", "Discovery Area"),
    }

    metrics = {
        "total_disturbed_area_m2": detection["total_disturbed_area_m2"],
        "delta_ndvi_mean": detection["delta_ndvi_mean"],
        "candidates_count": len(state["candidates"]),
    }

    vision_out = service.gemini_client.analyze_discovery_imagery(
        rgb_before_path=detection["rgb_before_path"],
        rgb_after_path=detection["rgb_after_path"],
        candidates=state["candidates"],
        metrics=metrics,
        scene_metadata=scene_metadata,
    )

    return {
        "vision_result": vision_out,
        "audit": vision_out.get("audit").model_dump() if hasattr(vision_out.get("audit"), "model_dump") else vision_out.get("audit"),
    }


def generate_georeferenced_report_node(state: DiscoveryWorkflowState) -> dict[str, Any]:
    """Assemble final DiscoveryReport and GeoJSON FeatureCollection."""
    if state.get("errors"):
        return {
            "report": {
                "discovery_id": state.get("discovery_id", "ERROR"),
                "status": "FAILED",
                "errors": state["errors"],
            }
        }

    service = ConstructionDiscoveryService()
    # Re-run full assemble to ensure clean serialization and GeoJSON saving
    report = service.run_discovery(
        latitude=state["latitude"],
        longitude=state["longitude"],
        radius_km=state["radius_km"],
        before_date_start=state["before_date_start"],
        before_date_end=state["before_date_end"],
        after_date_start=state["after_date_start"],
        after_date_end=state["after_date_end"],
        min_change_area_m2=state.get("min_change_area_m2", 5000.0),
        max_cloud_cover=state.get("max_cloud_cover", 20.0),
        area_name=state.get("area_name", "Independent Site Discovery"),
        discovery_id=state["discovery_id"],
    )

    return {
        "report": report.model_dump(),
        "audit": report.vision_audit.model_dump(),
    }


def create_construction_discovery_graph():
    """Build the LangGraph workflow for independent construction discovery."""
    workflow = StateGraph(DiscoveryWorkflowState)

    workflow.add_node("validate_area", validate_area_node)
    workflow.add_node("retrieve_before_after_imagery", retrieve_before_after_imagery_node)
    workflow.add_node("preprocess_and_align_images", preprocess_and_align_images_node)
    workflow.add_node("detect_candidate_changes", detect_candidate_changes_node)
    workflow.add_node("analyze_candidates_with_gemini_vision", analyze_candidates_with_gemini_vision_node)
    workflow.add_node("generate_georeferenced_report", generate_georeferenced_report_node)

    workflow.add_edge(START, "validate_area")
    workflow.add_edge("validate_area", "retrieve_before_after_imagery")
    workflow.add_edge("retrieve_before_after_imagery", "preprocess_and_align_images")
    workflow.add_edge("preprocess_and_align_images", "detect_candidate_changes")
    workflow.add_edge("detect_candidate_changes", "analyze_candidates_with_gemini_vision")
    workflow.add_edge("analyze_candidates_with_gemini_vision", "generate_georeferenced_report")
    workflow.add_edge("generate_georeferenced_report", END)

    return workflow.compile()


construction_discovery_agent = create_construction_discovery_graph()

