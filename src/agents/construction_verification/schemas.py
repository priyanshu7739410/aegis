from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class VerificationStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    VISION_NOT_EXECUTED = "VISION_NOT_EXECUTED"
    NOT_ANALYZED = "NOT_ANALYZED"


class OfficerDecisionType(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_INSPECTION = "REQUEST_INSPECTION"
    HOLD = "HOLD"


class ConstructionClaim(BaseModel):
    activity: str = Field(description="Name or description of the construction activity claimed.")
    claimed_status: str = Field(
        default="completed",
        description="Claimed completion status (e.g. completed, in progress, 100%)."
    )
    quantity: str | None = Field(
        default=None,
        description="Any specified dimensions, quantities, or volumes mentioned in the document."
    )
    details: str | None = Field(
        default=None,
        description="Additional technical or location specifics regarding the claim."
    )


class LocationInfo(BaseModel):
    site_name: str | None = Field(default=None, description="Name or identifier of the project site.")
    address: str | None = Field(default=None, description="Physical address or administrative area.")
    latitude: float | None = Field(default=None, description="WGS84 latitude coordinate in decimal degrees.")
    longitude: float | None = Field(default=None, description="WGS84 longitude coordinate in decimal degrees.")
    bbox: list[float] | None = Field(
        default=None,
        description="Bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84."
    )


class ExtractedProjectInfo(BaseModel):
    project_name: str | None = Field(default=None, description="Name or contract title of the construction project.")
    contractor: str | None = Field(default=None, description="Contractor named in the document.")
    construction_type: str | None = Field(
        default=None,
        description="Type of construction (e.g., Road construction, Building construction, Bridge, Pipeline).",
    )
    payment_amount: str | None = Field(
        default=None,
        description="Claimed payment amount with currency, exactly as stated in the document.",
    )
    location: LocationInfo | None = Field(default=None, description="Site location information.")
    start_date: str | None = Field(
        default=None,
        description="Start date of the work (YYYY-MM-DD or as stated in the document)."
    )
    completion_date: str | None = Field(
        default=None,
        description="Claimed completion date (YYYY-MM-DD or as stated in the document)."
    )
    claims: list[ConstructionClaim] = Field(
        default_factory=list,
        description="List of specific construction activities claimed to be completed."
    )
    document_notes: list[str] = Field(
        default_factory=list,
        description="Non-construction items, billing categories, date ranges, boilerplate, or administrative notes."
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Critical fields that could not be found in the document."
    )
    uncertain_fields: list[str] = Field(
        default_factory=list,
        description="Fields where extracted information has high ambiguity or uncertainty."
    )


class SceneMetadata(BaseModel):
    scene_id: str
    acquisition_date: str
    cloud_cover_percent: float
    site_cloud_cover_percent: float | None = None
    asset_urls: dict[str, str] = Field(default_factory=dict)
    local_rgb_path: str | None = None


class SatelliteMetrics(BaseModel):
    mean_ndvi_before: float = Field(default=0.0, description="Average NDVI value across the site prior to construction.")
    mean_ndvi_after: float = Field(default=0.0, description="Average NDVI value across the site after claimed construction.")
    delta_ndvi: float = Field(default=0.0, description="Difference in mean NDVI (after - before).")
    mean_bsi_before: float | None = Field(default=None, description="Average Bare Soil Index before.")
    mean_bsi_after: float | None = Field(default=None, description="Average Bare Soil Index after.")
    delta_bsi: float | None = Field(default=None, description="Difference in Bare Soil Index (after - before).")
    vegetation_loss_fraction: float = Field(
        default=0.0,
        description="Fraction of site area showing significant vegetation loss (e.g. delta_ndvi < -0.15)."
    )
    potential_disturbed_area_m2: float = Field(
        default=0.0,
        description="Estimated land disturbed area in square meters based on spectral changes."
    )
    valid_observation_coverage_pct: float = Field(
        default=0.0,
        description="Percentage of site pixels with cloud-free, high-quality data."
    )
    ndvi_diff_image_path: str | None = None
    rgb_before_preview_path: str | None = None
    rgb_after_preview_path: str | None = None
    before_scene_id: str | None = None
    after_scene_id: str | None = None
    crs: str | None = None
    raster_shape: list[int] | None = None
    pixel_area_m2: float | None = None
    synthetic: bool = False


class VisionClaimFinding(BaseModel):
    claim: str = Field(description="The construction claim assessed.")
    observed_changes: list[str] = Field(
        default_factory=list,
        description="Visible physical ground changes observed between the before and after satellite images."
    )
    supports_claim: bool | None | str = Field(
        default=None,
        description="True if visible evidence directly supports the observable claim, False if contradicted, None if inconclusive, or 'NOT_ANALYZED' / 'VISION_NOT_EXECUTED'."
    )
    reasoning: str = Field(description="Evidence-based rationale for the assessment.")
    limitations: list[str] = Field(
        default_factory=list,
        description="Satellite resolution, spectral, or temporal limitations (e.g. 10m Sentinel-2 pixel size, sub-surface invisibility)."
    )
    status: Literal["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE", "VISION_NOT_EXECUTED"] | None = None
    recommended_follow_up: str | None = None

    @model_validator(mode="after")
    def _derive_status(self) -> "VisionClaimFinding":
        if self.status is None:
            if self.supports_claim is True:
                self.status = "SUPPORTED"
            elif self.supports_claim is False:
                self.status = "NOT_SUPPORTED"
            elif self.supports_claim in ("NOT_ANALYZED", "VISION_NOT_EXECUTED"):
                self.status = "VISION_NOT_EXECUTED"
            else:
                self.status = "INSUFFICIENT_EVIDENCE"
        return self


class ClaimVisionAssessment(BaseModel):
    """Per-claim result requested from Gemini. Kept free of unions and audit fields."""

    claim: str = Field(description="The claim text, copied exactly as provided.")
    status: Literal["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE"] = Field(
        description="SUPPORTED if visible evidence supports the claim, NOT_SUPPORTED if contradicted, otherwise INSUFFICIENT_EVIDENCE."
    )
    visual_observations: list[str] = Field(default_factory=list, description="What is visible in the two images.")
    reasoning: str = Field(description="Evidence-based explanation of the status.")
    limitations: list[str] = Field(default_factory=list, description="What these images cannot establish.")
    recommended_follow_up: str = Field(description="Suggested next verification step for the human officer.")


class VisionModelResponse(BaseModel):
    """Structured output schema sent to Gemini Vision."""

    findings: list[ClaimVisionAssessment] = Field(default_factory=list)
    general_observations: str = ""


class VisionAnalysisOutput(BaseModel):
    findings: list[VisionClaimFinding] = Field(
        default_factory=list,
        description="Vision findings for each construction claim."
    )
    general_observations: str = Field(
        default="",
        description="General description of visible surface transformations on site."
    )
    vision_audit: "VisionExecutionAudit | None" = None


class ClaimVerificationResult(BaseModel):
    claim: str = Field(description="Claimed construction activity.")
    status: Literal["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE", "VISION_NOT_EXECUTED", "NOT_ANALYZED"] = Field(
        description="Verification outcome. SUPPORTED: visible evidence supports claim. NOT_SUPPORTED: contradictory evidence. INSUFFICIENT_EVIDENCE: inconclusive or requires higher resolution/on-site inspection. VISION_NOT_EXECUTED / NOT_ANALYZED: image data missing, vision not executed, or model rate-limited/failed."
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Physical and spectral observations from satellite or drone imagery."
    )
    reason: str = Field(description="Detailed explanation justifying the status.")
    limitations: list[str] = Field(
        default_factory=list,
        description="Physical, optical, or resolution constraints affecting the assessment."
    )
    recommended_next_action: str = Field(
        description="Next operational step (e.g., Approve payment claim, Request drone inspection, Request physical engineering test)."
    )


class DroneVerificationFinding(BaseModel):
    claim: str
    observations: list[str] = Field(default_factory=list)
    supports_claim: Literal["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE", "VISION_NOT_EXECUTED", "NOT_ANALYZED"] | None = None
    reasoning: str
    limitations: list[str] = Field(default_factory=list)


class DroneAnalysisOutput(BaseModel):
    findings: list[DroneVerificationFinding] = Field(default_factory=list)
    general_observations: str | None = None


class OfficerDecisionRecord(BaseModel):
    case_id: str
    decision: Literal["APPROVE", "REJECT", "REQUEST_INSPECTION", "HOLD"]
    officer_name_or_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str | None = None
    justification: str | None = None
    report_sha256: str | None = None
    evidence_mode: str | None = None


class VerificationReport(BaseModel):
    case_id: str
    project_name: str
    construction_type: str
    contractor: str | None = None
    payment_amount: str | None = None
    evidence_mode: Literal["LIVE_SERVICES", "SYNTHETIC_TEST_DATA"] = "LIVE_SERVICES"
    before_scene: dict[str, Any] | None = None
    after_scene: dict[str, Any] | None = None
    vision_audit: dict[str, Any] | None = None
    general_observations: str | None = None
    evidence_limitations: list[str] = Field(default_factory=list)
    location: dict[str, Any]
    start_date: str | None = None
    completion_date: str | None = None
    overall_status: str
    claims_summary: list[ClaimVerificationResult] = Field(default_factory=list)
    satellite_metrics: SatelliteMetrics | None = None
    requires_drone_verification: bool = False
    drone_inspection_request: dict[str, Any] | None = None
    drone_findings: list[DroneVerificationFinding] | None = None
    officer_decision: OfficerDecisionRecord | None = None
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    audit_log: list[str] = Field(default_factory=list)


class VisionExecutionAudit(BaseModel):
    """Instrumentation recording exact parameters and proof of multimodal vision processing."""
    before_scene_id: str | None = None
    after_scene_id: str | None = None
    before_acquisition_date: str | None = None
    after_acquisition_date: str | None = None
    raster_dimensions: tuple[int, int] | None = None
    source_resolution_m: float = 10.0
    rgb_image_dimensions: tuple[int, int] | None = None
    image_mime_types: list[str] = Field(default_factory=lambda: ["image/png", "image/png"])
    image_byte_lengths: list[int] = Field(default_factory=list)
    both_images_included: bool = False
    gemini_model_id: str | None = None
    model_used: str | None = None
    api_call_status: str = "NOT_ATTEMPTED"  # "SUCCESS", "FAILED", "NOT_ATTEMPTED"
    error_message: str | None = None
    parsed_response_status: str = "PENDING"  # "SUCCESS", "FAILED", "VISION_NOT_EXECUTED", "INSUFFICIENT_EVIDENCE"
    vision_actually_executed: bool = False
    execution_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CandidateClassification(StrEnum):
    CONSTRUCTION = "CONSTRUCTION"
    AGRICULTURAL_HARVEST = "AGRICULTURAL_HARVEST"
    WATER_BODY = "WATER_BODY"
    CLOUD_OR_ARTIFACT = "CLOUD_OR_ARTIFACT"
    UNCLEAR = "UNCLEAR"


class DiscoveryCandidateFinding(BaseModel):
    candidate_id: str
    classification: CandidateClassification
    confidence: float = Field(ge=0.0, le=1.0)
    visual_evidence: str
    limitations: str = ""


class DiscoveryVisionStructuredOutput(BaseModel):
    overall_summary: str = Field(description="Summary of overall physical developments across the area")
    candidate_findings: list[DiscoveryCandidateFinding] = Field(default_factory=list)


class DiscoveryCandidate(BaseModel):
    """Georeferenced candidate area with identified land-cover or construction change."""
    candidate_id: str
    centroid: dict[str, float] = Field(description="Geographic center {'latitude': lat, 'longitude': lon}")
    bbox: list[float] = Field(description="[min_lon, min_lat, max_lon, max_lat] in WGS84")
    area_m2: float
    area_hectares: float
    mean_delta_ndvi: float
    mean_delta_bsi: float
    change_type: str = "Surface Disturbance / Construction Candidate"
    geometry: dict[str, Any] = Field(description="GeoJSON geometry dictionary in EPSG:4326")
    classification: CandidateClassification | None = None
    confidence: float | None = Field(default=None, description="Confidence score if computed or evaluated by model")
    visual_evidence: str | None = None
    limitations: str | None = None
    gemini_interpretation: str | None = None


class DiscoveryReport(BaseModel):
    """Complete georeferenced report of independent construction discovery."""
    discovery_id: str
    area_name: str
    center: dict[str, float]
    radius_km: float
    bounding_box: list[float]
    before_scene_id: str
    before_acquisition_date: str
    before_cloud_cover_pct: float
    after_scene_id: str
    after_acquisition_date: str
    after_cloud_cover_pct: float
    crs: str
    raster_shape: list[int]
    resolution_m: float
    total_disturbed_area_m2: float
    total_disturbed_area_hectares: float
    detected_change_area_m2: float = 0.0
    detected_change_area_hectares: float = 0.0
    raw_change_area_m2: float = 0.0
    raw_change_area_hectares: float = 0.0
    top_candidates_area_m2: float = 0.0
    top_candidates_area_hectares: float = 0.0
    top_candidates_fraction_of_raw: float = 0.0
    runtime_breakdown: dict[str, float] = Field(default_factory=dict)
    total_runtime_seconds: float = 0.0
    classified_construction_area_hectares: float = 0.0
    classified_agricultural_area_hectares: float = 0.0
    classified_other_area_hectares: float = 0.0
    persistent_change_area_m2: float | None = None
    persistent_change_area_hectares: float | None = None
    seasonal_vegetation_area_m2: float | None = None
    confirmation_scene_id: str | None = None
    confirmation_acquisition_date: str | None = None
    confirmation_cloud_cover_pct: float | None = None
    seasonality_note: str = "2-scene analysis: cannot distinguish permanent construction from seasonal crop cycles without a confirmation scene. Use caution with agricultural areas."
    retrieval_attempts: list[dict[str, Any]] = Field(default_factory=list)
    candidates_count: int
    candidates: list[DiscoveryCandidate] = Field(default_factory=list)
    gemini_vision_summary: str
    vision_audit: VisionExecutionAudit
    geojson_feature_collection: dict[str, Any] = Field(default_factory=dict)
    rgb_before_path: str | None = None
    rgb_after_path: str | None = None
    change_mask_path: str | None = None
    ndvi_diff_path: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

