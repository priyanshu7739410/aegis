import asyncio
import concurrent.futures
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from agents.construction_verification.schemas import (
    DiscoveryReport,
    OfficerDecisionRecord,
    OfficerDecisionType,
)
from agents.construction_verification.services.discovery_service import (
    ConstructionDiscoveryService,
    DiscoveryError,
)
from agents.construction_verification.services.ocr_client import OCRClient, OCRError
from core.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/construction", tags=["Construction Verification"])

UPLOAD_DIR = Path(settings.CONSTRUCTION_OUTPUT_DIR) / "uploads"
DECISIONS_DIR = Path(settings.CONSTRUCTION_OUTPUT_DIR) / "decisions"
CASES_DIR = Path(settings.CONSTRUCTION_OUTPUT_DIR) / "cases"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
CASES_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
DISCOVERY_JOBS: dict[str, dict[str, Any]] = {}


class UploadResponse(BaseModel):
    file_id: str
    file_name: str
    file_path: str
    case_id: str
    extracted_text: str | None = None
    ocr_status: str = "SKIPPED"  # "SUCCESS", "FAILED", "SKIPPED"
    ocr_error: str | None = None
    file_type: str


class CaseUpdateRequest(BaseModel):
    extracted_text: str | None = None
    project_name: str | None = None
    construction_type: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    start_date: str | None = None
    completion_date: str | None = None
    claims: list[dict[str, Any]] | None = None
    manual_corrections: dict[str, Any] | None = None


class CaseDecisionInput(BaseModel):
    case_id: str
    decision: OfficerDecisionType
    officer_name_or_id: str = Field(description="Officer identifier")
    notes: str | None = None


class CaseDecisionResponse(BaseModel):
    status: str
    decision_record: OfficerDecisionRecord
    message: str


def _get_case_file(case_id: str) -> Path:
    safe_id = "".join(c for c in case_id if c.isalnum() or c in ("-", "_")).strip()
    return CASES_DIR / f"{safe_id}.json"


def load_case_record(case_id: str) -> dict[str, Any]:
    file_path = _get_case_file(case_id)
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading case file {file_path}: {e}")
    return {
        "case_id": case_id,
        "document_paths": [],
        "document_name": None,
        "extracted_text": "",
        "ocr_status": "idle",
        "ocr_error": None,
        "project_name": "",
        "construction_type": "",
        "construction_location": {},
        "bounding_box": None,
        "start_date": "",
        "completion_date": "",
        "claims": [],
        "manual_corrections": {},
        "satellite_metrics": None,
        "verification_results": None,
        "drone_image_paths": [],
        "errors": [],
    }


def save_case_record(case_id: str, data: dict[str, Any]) -> None:
    file_path = _get_case_file(case_id)
    data["case_id"] = case_id
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to persist case record to {file_path}: {e}")


async def sync_to_checkpointer(case_id: str, values: dict[str, Any]) -> None:
    try:
        from agents.agents import get_agent
        agent = get_agent("construction-verification")
        if agent and hasattr(agent, "checkpointer") and agent.checkpointer:
            config = {"configurable": {"thread_id": case_id}}
            await agent.aupdate_state(config, values)
            logger.info(f"Synchronized case {case_id} to LangGraph checkpointer.")
    except Exception as e:
        logger.debug(f"LangGraph checkpointer sync skipped/failed for case {case_id}: {e}")


@router.post("/upload", response_model=UploadResponse)
async def upload_construction_file(
    file: UploadFile = File(...),
    case_id: str | None = Form(None),
    run_ocr: bool = Form(True),
) -> UploadResponse:
    """Upload construction document (PDF/Image) or drone photography with durable case association."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing in upload.")

    suffix = Path(file.filename).suffix.lower()
    allowed_suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".txt"}
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{suffix}' not supported. Allowed: {', '.join(allowed_suffixes)}",
        )

    assigned_case_id = (case_id or "").strip()
    if not assigned_case_id:
        assigned_case_id = f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    file_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    safe_name = f"{file_id}_{Path(file.filename).name}"
    target_path = UPLOAD_DIR / safe_name

    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds size limit of {MAX_UPLOAD_SIZE // (1024*1024)}MB.",
            )

        with open(target_path, "wb") as f:
            f.write(content)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from e

    extracted_text = None
    ocr_status_str = "SKIPPED"
    ocr_error_msg = None

    if run_ocr and suffix in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"}:
        try:
            client = OCRClient()
            extracted_text = await client.extract_text_from_file(target_path)
            ocr_status_str = "SUCCESS"
        except OCRError as e:
            logger.warning(f"OCR failed for uploaded file {safe_name}: {e}")
            ocr_status_str = "FAILED"
            ocr_error_msg = str(e)
        except Exception as e:
            logger.warning(f"Unexpected OCR failure for {safe_name}: {e}")
            ocr_status_str = "FAILED"
            ocr_error_msg = str(e)

    file_type = "drone_image" if suffix in {".jpg", ".jpeg", ".png"} and not run_ocr else "document"

    # Persist in case record
    case_record = load_case_record(assigned_case_id)
    doc_path_resolved = str(target_path.resolve())

    if file_type == "drone_image":
        if doc_path_resolved not in case_record.setdefault("drone_image_paths", []):
            case_record["drone_image_paths"].append(doc_path_resolved)
    else:
        if doc_path_resolved not in case_record.setdefault("document_paths", []):
            case_record["document_paths"].append(doc_path_resolved)
        case_record["document_name"] = file.filename
        if extracted_text:
            case_record["extracted_text"] = extracted_text
        case_record["ocr_status"] = ocr_status_str
        case_record["ocr_error"] = ocr_error_msg

    save_case_record(assigned_case_id, case_record)

    # Sync to LangGraph checkpointer
    checkpoint_vals: dict[str, Any] = {
        "case_id": assigned_case_id,
        "document_paths": case_record.get("document_paths", []),
    }
    if extracted_text:
        checkpoint_vals["extracted_text"] = extracted_text
    if file_type == "drone_image":
        checkpoint_vals["drone_image_paths"] = case_record.get("drone_image_paths", [])

    await sync_to_checkpointer(assigned_case_id, checkpoint_vals)

    return UploadResponse(
        file_id=file_id,
        file_name=file.filename,
        file_path=doc_path_resolved,
        case_id=assigned_case_id,
        extracted_text=extracted_text,
        ocr_status=ocr_status_str,
        ocr_error=ocr_error_msg,
        file_type=file_type,
    )


@router.get("/cases/{case_id}")
@router.get("/case_state/{case_id}")
async def get_case_state(case_id: str) -> dict[str, Any]:
    """Retrieve durable case state including documents, OCR, extracted claims, and verification findings."""
    case_file = _get_case_file(case_id)
    if not case_file.exists():
        raise HTTPException(status_code=404, detail=f"No case found with ID {case_id}")
    return load_case_record(case_id)


@router.post("/cases/{case_id}/update")
async def update_case_state(case_id: str, update: CaseUpdateRequest) -> dict[str, Any]:
    """Update case state with officer edits, normalized coordinates, and date corrections."""
    case_record = load_case_record(case_id)
    checkpoint_vals: dict[str, Any] = {"case_id": case_id}

    if update.extracted_text is not None:
        case_record["extracted_text"] = update.extracted_text
        checkpoint_vals["extracted_text"] = update.extracted_text

    if update.project_name is not None:
        case_record["project_name"] = update.project_name
        checkpoint_vals["project_name"] = update.project_name

    if update.construction_type is not None:
        case_record["construction_type"] = update.construction_type
        checkpoint_vals["construction_type"] = update.construction_type

    # Validate and normalize coordinates
    lat = update.latitude
    lon = update.longitude
    if lat is not None and not (-90.0 <= lat <= 90.0):
        raise HTTPException(status_code=422, detail="Latitude must be between -90.0 and 90.0")
    if lon is not None and not (-180.0 <= lon <= 180.0):
        raise HTTPException(status_code=422, detail="Longitude must be between -180.0 and 180.0")

    if lat is not None and lon is not None:
        bbox = [
            round(lon - 0.005, 6),
            round(lat - 0.005, 6),
            round(lon + 0.005, 6),
            round(lat + 0.005, 6),
        ]
        location_dict = {
            "site_name": update.project_name or case_record.get("project_name", ""),
            "latitude": lat,
            "longitude": lon,
            "bbox": bbox,
            "preliminary_search_location": True,
        }
        case_record["construction_location"] = location_dict
        case_record["bounding_box"] = bbox
        checkpoint_vals["construction_location"] = location_dict
        checkpoint_vals["bounding_box"] = bbox

    # Normalize dates
    if update.start_date is not None:
        norm_start = update.start_date.strip().replace("/", "-")
        case_record["start_date"] = norm_start
        checkpoint_vals["start_date"] = norm_start

    if update.completion_date is not None:
        norm_comp = update.completion_date.strip().replace("/", "-")
        case_record["completion_date"] = norm_comp
        checkpoint_vals["completion_date"] = norm_comp

    if update.claims is not None:
        case_record["claims"] = update.claims
        checkpoint_vals["construction_claims"] = update.claims

    if update.manual_corrections:
        case_record.setdefault("manual_corrections", {}).update(update.manual_corrections)
        checkpoint_vals["manual_corrections"] = case_record["manual_corrections"]

    save_case_record(case_id, case_record)
    await sync_to_checkpointer(case_id, checkpoint_vals)

    return {"status": "UPDATED", "case_id": case_id, "case_data": case_record}


@router.post("/cases/{case_id}/verify")
async def run_case_verification(case_id: str) -> dict[str, Any]:
    """Execute LangGraph verification workflow for a validated case."""
    from agents.agents import get_agent

    case_record = load_case_record(case_id)
    agent = get_agent("construction-verification")
    if not agent:
        raise HTTPException(status_code=500, detail="construction-verification agent is not registered.")

    # Validate prerequisites
    bbox = case_record.get("bounding_box")
    start_date = case_record.get("start_date")
    completion_date = case_record.get("completion_date")

    if not bbox or not start_date or not completion_date:
        missing = []
        if not bbox:
            missing.append("Coordinates (latitude, longitude)")
        if not start_date:
            missing.append("Start date")
        if not completion_date:
            missing.append("Completion date")
        raise HTTPException(
            status_code=400,
            detail=f"Cannot run verification. Missing required case data: {', '.join(missing)}",
        )

    inputs: dict[str, Any] = {
        "case_id": case_id,
        "document_paths": case_record.get("document_paths", []),
        "extracted_text": case_record.get("extracted_text", ""),
        "project_name": case_record.get("project_name", ""),
        "construction_type": case_record.get("construction_type", ""),
        "construction_location": case_record.get("construction_location", {}),
        "bounding_box": bbox,
        "start_date": start_date,
        "completion_date": completion_date,
        "construction_claims": case_record.get("claims", []),
        "manual_corrections": case_record.get("manual_corrections", {}),
    }

    try:
        config = {"configurable": {"thread_id": case_id}}
        result = await agent.ainvoke(inputs, config=config)

        # Update case record with results
        if "satellite_metrics" in result and result["satellite_metrics"]:
            case_record["satellite_metrics"] = result["satellite_metrics"]
        if "satellite_error" in result and result["satellite_error"]:
            case_record["satellite_error"] = result["satellite_error"]
        if "before_scene" in result and result["before_scene"]:
            case_record["before_scene"] = result["before_scene"]
        if "after_scene" in result and result["after_scene"]:
            case_record["after_scene"] = result["after_scene"]
        if "verification_results" in result and result["verification_results"]:
            case_record["verification_results"] = result["verification_results"]
        if "overall_status" in result:
            case_record["overall_status"] = result["overall_status"]
        if "errors" in result:
            case_record["errors"] = result["errors"]

        save_case_record(case_id, case_record)
        return {
            "status": "COMPLETED",
            "case_id": case_id,
            "verification_results": case_record.get("verification_results"),
            "satellite_metrics": case_record.get("satellite_metrics"),
            "overall_status": case_record.get("overall_status"),
            "errors": case_record.get("errors", []),
        }
    except Exception as e:
        logger.error(f"Verification pipeline failed for case {case_id}: {e}")
        case_record.setdefault("errors", []).append(str(e))
        save_case_record(case_id, case_record)
        raise HTTPException(status_code=500, detail=f"Verification failed: {e}") from e


@router.post("/decision", response_model=CaseDecisionResponse)
async def record_officer_decision(
    decision_input: CaseDecisionInput,
) -> CaseDecisionResponse:
    """Record an official human officer decision for payment approval, rejection, or inspection request.

    SECURITY RULE: AI cannot approve payments. Only an authorized human officer may record this decision.
    """
    record = OfficerDecisionRecord(
        case_id=decision_input.case_id,
        decision=decision_input.decision.value,
        officer_name_or_id=decision_input.officer_name_or_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        notes=decision_input.notes,
    )

    case_file = DECISIONS_DIR / f"{decision_input.case_id}_decision.json"
    try:
        with open(case_file, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json(indent=2))
    except Exception as e:
        logger.error(f"Failed to record officer decision: {e}")
        raise HTTPException(status_code=500, detail="Failed to persist decision audit record.")

    # Also record in case file
    case_record = load_case_record(decision_input.case_id)
    case_record["officer_decision"] = record.model_dump()
    save_case_record(decision_input.case_id, case_record)

    action_text = {
        OfficerDecisionType.APPROVE: "Final payment approved by officer.",
        OfficerDecisionType.REJECT: "Payment claim rejected by officer.",
        OfficerDecisionType.REQUEST_INSPECTION: "Case held for drone or physical engineering inspection.",
        OfficerDecisionType.HOLD: "Case placed on administrative hold.",
    }.get(decision_input.decision, "Decision recorded.")

    return CaseDecisionResponse(
        status="RECORDED",
        decision_record=record,
        message=action_text,
    )


@router.get("/case/{case_id}")
async def get_case_decision(case_id: str) -> dict:
    """Retrieve decision record and audit trail for a case."""
    case_file = DECISIONS_DIR / f"{case_id}_decision.json"
    if not case_file.exists():
        # Fallback check in case record
        case_record = load_case_record(case_id)
        if case_record.get("officer_decision"):
            return case_record["officer_decision"]
        raise HTTPException(status_code=404, detail=f"No decision record found for case {case_id}")

    try:
        with open(case_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading decision for case {case_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not read decision record.")


class DiscoveryInput(BaseModel):
    latitude: float = Field(default=28.17, description="Centroid latitude")
    longitude: float = Field(default=77.61, description="Centroid longitude")
    radius_km: float = Field(default=5.0, description="Search radius in kilometers")
    before_date_start: str = Field(default="2021-01-01", description="Pre-construction start date (YYYY-MM-DD)")
    before_date_end: str = Field(default="2021-03-31", description="Pre-construction end date (YYYY-MM-DD)")
    after_date_start: str = Field(default="2025-01-01", description="Post-construction start date (YYYY-MM-DD)")
    after_date_end: str = Field(default="2025-03-31", description="Post-construction end date (YYYY-MM-DD)")
    confirmation_date_start: str | None = Field(default=None, description="Optional confirmation start date (YYYY-MM-DD)")
    confirmation_date_end: str | None = Field(default=None, description="Optional confirmation end date (YYYY-MM-DD)")
    min_change_area_m2: float = Field(default=5000.0, description="Minimum candidate mapping area in m^2")
    max_cloud_cover: float = Field(default=20.0, description="Maximum scene cloud cover percentage")
    area_name: str = Field(default="Noida International Airport, Jewar", description="Descriptive name")


class DiscoveryJobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: int = 0
    message: str = "Job queued"
    discovery_id: str | None = None
    created_at: str
    updated_at: str
    report: dict[str, Any] | None = None
    error: str | None = None


def _execute_discovery_sync(job_id: str, input_data: DiscoveryInput):
    """Worker function executed in thread pool for asynchronous satellite discovery."""
    service = ConstructionDiscoveryService()

    def update_progress(pct: int, msg: str):
        if job_id in DISCOVERY_JOBS:
            DISCOVERY_JOBS[job_id]["progress"] = pct
            DISCOVERY_JOBS[job_id]["message"] = msg
            DISCOVERY_JOBS[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        DISCOVERY_JOBS[job_id]["status"] = "processing"
        DISCOVERY_JOBS[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        report = service.run_discovery(
            latitude=input_data.latitude,
            longitude=input_data.longitude,
            radius_km=input_data.radius_km,
            before_date_start=input_data.before_date_start,
            before_date_end=input_data.before_date_end,
            after_date_start=input_data.after_date_start,
            after_date_end=input_data.after_date_end,
            confirmation_date_start=input_data.confirmation_date_start,
            confirmation_date_end=input_data.confirmation_date_end,
            min_change_area_m2=input_data.min_change_area_m2,
            max_cloud_cover=input_data.max_cloud_cover,
            area_name=input_data.area_name,
            progress_callback=update_progress,
        )
        DISCOVERY_JOBS[job_id]["status"] = "completed"
        DISCOVERY_JOBS[job_id]["progress"] = 100
        DISCOVERY_JOBS[job_id]["message"] = f"Discovery completed. Found {len(report.candidates)} candidate zones."
        DISCOVERY_JOBS[job_id]["discovery_id"] = report.discovery_id
        DISCOVERY_JOBS[job_id]["report"] = report.model_dump()
        DISCOVERY_JOBS[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        logger.error(f"Discovery job {job_id} failed: {e}")
        DISCOVERY_JOBS[job_id]["status"] = "failed"
        DISCOVERY_JOBS[job_id]["message"] = f"Discovery pipeline failed: {e}"
        DISCOVERY_JOBS[job_id]["error"] = str(e)
        DISCOVERY_JOBS[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


@router.post("/discover", response_model=DiscoveryJobStatus, status_code=status.HTTP_202_ACCEPTED)
async def run_construction_discovery(input_data: DiscoveryInput) -> dict[str, Any]:
    """Start an asynchronous independent satellite construction discovery job (Mode B)."""
    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    now_str = datetime.now(timezone.utc).isoformat()
    job_info = {
        "job_id": job_id,
        "status": "processing",
        "progress": 5,
        "message": f"Queued discovery for {input_data.area_name}...",
        "discovery_id": None,
        "created_at": now_str,
        "updated_at": now_str,
        "report": None,
        "error": None,
    }
    DISCOVERY_JOBS[job_id] = job_info

    loop = asyncio.get_running_loop()
    loop.run_in_executor(_thread_pool, _execute_discovery_sync, job_id, input_data)
    return job_info


@router.get("/discover/{id}/status", response_model=DiscoveryJobStatus)
async def get_discovery_job_status(id: str) -> dict[str, Any]:
    """Poll asynchronous discovery job progress or retrieve completed job status."""
    if id in DISCOVERY_JOBS:
        return DISCOVERY_JOBS[id]

    # Support looking up completed discovery ID on disk
    service = ConstructionDiscoveryService()
    report_file = service.output_dir / id / "discovery_report.json"
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
            now_str = datetime.now(timezone.utc).isoformat()
            return {
                "job_id": id,
                "status": "completed",
                "progress": 100,
                "message": "Discovery completed",
                "discovery_id": id,
                "created_at": now_str,
                "updated_at": now_str,
                "report": rep_data,
                "error": None,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not read discovery report: {e}")

    raise HTTPException(status_code=404, detail=f"Discovery job or ID '{id}' not found")


@router.get("/discover/{discovery_id}")
async def get_discovery_report(discovery_id: str) -> dict[str, Any]:
    """Retrieve past discovery run report."""
    service = ConstructionDiscoveryService()
    report_file = service.output_dir / discovery_id / "discovery_report.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail=f"No discovery report found for ID {discovery_id}")
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read discovery report: {e}")


@router.get("/discover/{discovery_id}/geojson")
async def get_discovery_geojson(discovery_id: str) -> dict[str, Any]:
    """Download GeoJSON FeatureCollection of candidate construction polygons."""
    service = ConstructionDiscoveryService()
    geojson_file = service.output_dir / discovery_id / "candidates.geojson"
    if not geojson_file.exists():
        raise HTTPException(status_code=404, detail=f"No GeoJSON found for discovery ID {discovery_id}")
    try:
        with open(geojson_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read GeoJSON: {e}")

