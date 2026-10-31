from typing import Any
from langgraph.graph import MessagesState


class ConstructionVerificationState(MessagesState, total=False):
    """LangGraph state for the Construction Verification and Payment Approval agent."""

    case_id: str

    document_paths: list[str]
    extracted_text: str

    project_name: str
    construction_location: dict[str, Any]
    construction_type: str
    construction_claims: list[dict[str, Any]]

    start_date: str
    completion_date: str

    site_polygon: dict[str, Any]
    bounding_box: list[float]  # [min_lon, min_lat, max_lon, max_lat] in WGS84

    before_scene: dict[str, Any]
    after_scene: dict[str, Any]

    satellite_metrics: dict[str, Any]
    satellite_error: str | None
    vision_findings: dict[str, Any]
    vision_audit: dict[str, Any]

    verification_results: list[dict[str, Any]]
    overall_status: str

    requires_drone_verification: bool
    drone_image_paths: list[str]
    drone_findings: list[dict[str, Any]]

    officer_decision: dict[str, Any] | str
    report: dict[str, Any]

    manual_corrections: dict[str, Any]
    errors: list[str]

