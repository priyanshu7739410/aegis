"""Stage-by-stage case workflow behind src/construction_app.py.

A case is a plain dict holding LangGraph state plus UI bookkeeping. Each stage runs the
existing nodes through a small compiled LangGraph, so officer review can happen between
stages. This module has no Streamlit dependency.
"""

import asyncio
import hashlib
import io
import json
import logging
import re
import uuid
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, get_type_hints

from langgraph.graph import END, START, StateGraph
from PIL import Image, UnidentifiedImageError

from agents.construction_verification.nodes.extract_claims import extract_claims_node
from agents.construction_verification.nodes.ocr import ocr_node
from agents.construction_verification.nodes.satellite_analysis import satellite_analysis_node
from agents.construction_verification.nodes.satellite_search import satellite_search_node
from agents.construction_verification.nodes.verification import verification_node
from agents.construction_verification.nodes.vision_analysis import vision_analysis_node
from agents.construction_verification.schemas import OfficerDecisionRecord, OfficerDecisionType
from agents.construction_verification.state import ConstructionVerificationState
from core.settings import settings

logger = logging.getLogger(__name__)

ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
BBOX_HALF_WIDTH_DEG = 0.005

_NODES = {
    "ocr": ocr_node,
    "extract": extract_claims_node,
    "retrieve": satellite_search_node,
    "raster": satellite_analysis_node,
    "vision": vision_analysis_node,
    "report": verification_node,
}
_STATE_KEYS = set(get_type_hints(ConstructionVerificationState))

_EXTRACTION_KEYS = (
    "project_name", "contractor", "construction_type", "payment_amount", "construction_location",
    "bounding_box", "start_date", "completion_date", "construction_claims", "extraction_error",
    "extraction_ran", "document_notes", "review_confirmed",
)
_SCENE_KEYS = ("before_scene", "after_scene")
_RASTER_KEYS = ("satellite_metrics", "satellite_error")
_VISION_KEYS = ("vision_findings", "vision_audit")
_REPORT_KEYS = ("verification_results", "overall_status", "requires_drone_verification", "report")


class DecisionError(ValueError):
    """An officer decision was incomplete or not allowed for this case."""


def new_case() -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return {
        "case_id": f"CASE-{stamp}-{uuid.uuid4().hex[:8].upper()}",
        "document_paths": [],
        "form_revision": 0,
        "stage_errors": {},
    }


def _clear(case: dict[str, Any], *groups: tuple[str, ...]) -> None:
    for group in groups:
        for key in group:
            case.pop(key, None)


def attach_document(case: dict[str, Any], name: str, content: bytes) -> bool:
    """Store an upload and reset everything derived from any earlier document.

    Returns False when these exact bytes are already attached, which is what happens on
    every Streamlit rerun while the file stays in the uploader.
    """
    digest = hashlib.sha256(content).hexdigest()
    if case.get("document_sha256") == digest:
        return False

    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}'. Upload a PDF, PNG, JPG or JPEG.")
    if suffix != ".pdf":
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ValueError(f"'{name}' is not a readable image file.") from e

    upload_dir = Path(settings.CONSTRUCTION_OUTPUT_DIR) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{case['case_id']}_{digest[:12]}{suffix}"
    path.write_bytes(content)

    _clear(case, _EXTRACTION_KEYS, _SCENE_KEYS, _RASTER_KEYS, _VISION_KEYS, _REPORT_KEYS)
    case.pop("extracted_text", None)
    case.pop("ocr_error", None)
    case["stage_errors"] = {}
    case.update(document_paths=[str(path)], document_name=Path(name).name, document_sha256=digest)
    return True


def stage_complete(case: dict[str, Any], stage: str) -> bool:
    if stage == "ocr":
        return bool(case.get("extracted_text"))
    if stage == "extract":
        return bool(case.get("extraction_ran"))
    if stage == "retrieve":
        return bool(case.get("before_scene") and case.get("after_scene"))
    if stage == "analyze":
        return bool(
            case.get("report") and (case.get("vision_audit") or {}).get("vision_actually_executed")
        )
    if stage == "report":
        return bool(case.get("report"))
    raise ValueError(f"Unknown stage: {stage}")


def _prerequisite_problem(case: dict[str, Any], stage: str) -> str | None:
    if stage == "ocr" and not case.get("document_paths"):
        return "Upload a document first."
    if stage == "extract" and not case.get("extracted_text"):
        return "Document text is required before extraction."
    if stage == "retrieve" and not case.get("review_confirmed"):
        return "The officer must confirm the project details before satellite retrieval."
    if stage == "analyze" and not stage_complete(case, "retrieve"):
        return "Both a before scene and an after scene are required before analysis."
    return None


@lru_cache(maxsize=None)
def _compiled(nodes: tuple[str, ...]):
    graph = StateGraph(ConstructionVerificationState)
    previous = START
    for name in nodes:
        graph.add_node(name, _NODES[name])
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph.compile()


def _nodes_for(case: dict[str, Any], stage: str, force: bool) -> tuple[str, ...]:
    if stage == "analyze":
        # Measurements stay valid after a vision failure, so only vision is retried.
        nodes = []
        if force or not case.get("satellite_metrics"):
            nodes.append("raster")
        if force or not (case.get("vision_audit") or {}).get("vision_actually_executed"):
            nodes.append("vision")
        nodes.append("report")
        return tuple(nodes)
    return (stage,)


def _invalidate_before_run(case: dict[str, Any], stage: str, force: bool) -> None:
    if stage == "ocr":
        _clear(case, _EXTRACTION_KEYS, _SCENE_KEYS, _RASTER_KEYS, _VISION_KEYS, _REPORT_KEYS)
        case.pop("extracted_text", None)
    elif stage == "extract":
        _clear(case, _EXTRACTION_KEYS, _SCENE_KEYS, _RASTER_KEYS, _VISION_KEYS, _REPORT_KEYS)
    elif stage == "retrieve":
        _clear(case, _SCENE_KEYS, _RASTER_KEYS, _VISION_KEYS, _REPORT_KEYS)
    elif stage == "analyze":
        _clear(case, _REPORT_KEYS)
        if force:
            _clear(case, _RASTER_KEYS, _VISION_KEYS)


def run_stage(case: dict[str, Any], stage: str, force: bool = False) -> list[str]:
    """Run one stage and merge the result into the case. Returns the stage's errors.

    Completed stages are not repeated unless forced, so reruns cost no API calls.
    """
    if stage_complete(case, stage) and not force:
        return case["stage_errors"].get(stage, [])
    if case.get("officer_decision"):
        return ["This case already has a recorded decision. Start a new case to re-assess it."]

    problem = _prerequisite_problem(case, stage)
    if problem:
        case["stage_errors"][stage] = [problem]
        return [problem]

    _invalidate_before_run(case, stage, force)
    state = {key: case[key] for key in _STATE_KEYS if key in case}
    state.update(messages=[], errors=[])

    try:
        result = asyncio.run(_compiled(_nodes_for(case, stage, force)).ainvoke(state))
    except Exception as e:
        logger.exception("Stage %s failed for %s", stage, case["case_id"])
        case["stage_errors"][stage] = [f"{stage} failed: {e}"]
        return case["stage_errors"][stage]

    errors = list(result.get("errors") or [])
    result.pop("messages", None)
    result.pop("errors", None)
    case.update(result)
    case["stage_errors"][stage] = errors

    if stage == "extract":
        case["extraction_ran"] = True
        case["form_revision"] += 1
    if stage in ("analyze", "report"):
        _save_report(case)
    return errors


def _parse_coordinate(raw: Any, label: str, limit: float, errors: list[str]) -> float | None:
    text = str(raw if raw is not None else "").strip()
    if not text:
        errors.append(f"{label} is required.")
        return None
    try:
        value = float(text)
    except ValueError:
        errors.append(f"{label} must be a decimal number.")
        return None
    if not -limit <= value <= limit:
        errors.append(f"{label} must be between -{limit:g} and {limit:g}.")
        return None
    return value


def _parse_date(raw: Any, label: str, errors: list[str]) -> str | None:
    text = str(raw or "").strip().replace("/", "-")
    if not text:
        errors.append(f"{label} is required.")
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        errors.append(f"{label} must be a valid date in YYYY-MM-DD format.")
        return None


def review_form_values(case: dict[str, Any]) -> dict[str, str]:
    """Current case values as text for the review form. Missing values stay blank."""
    location = case.get("construction_location") or {}

    def text(value: Any) -> str:
        return "" if value is None else str(value)

    return {
        "project_name": text(case.get("project_name")),
        "contractor": text(case.get("contractor")),
        "construction_type": text(case.get("construction_type")),
        "latitude": text(location.get("latitude")),
        "longitude": text(location.get("longitude")),
        "start_date": text(case.get("start_date")),
        "completion_date": text(case.get("completion_date")),
        "payment_amount": text(case.get("payment_amount")),
        "claims": "\n".join(c.get("activity", "") for c in case.get("construction_claims") or []),
    }


def missing_review_fields(case: dict[str, Any]) -> list[str]:
    values = review_form_values(case)
    labels = {
        "latitude": "Latitude",
        "longitude": "Longitude",
        "start_date": "Start date",
        "completion_date": "Claimed completion date",
        "claims": "Construction claims",
    }
    return [label for key, label in labels.items() if not values[key].strip()]


def apply_officer_review(case: dict[str, Any], form: dict[str, Any]) -> list[str]:
    """Validate officer edits, write them to case state and invalidate stale results."""
    if case.get("officer_decision"):
        return ["This case already has a recorded decision. Start a new case to change it."]

    errors: list[str] = []
    latitude = _parse_coordinate(form.get("latitude"), "Latitude", 90, errors)
    longitude = _parse_coordinate(form.get("longitude"), "Longitude", 180, errors)
    start_date = _parse_date(form.get("start_date"), "Start date", errors)
    completion_date = _parse_date(form.get("completion_date"), "Claimed completion date", errors)
    if start_date and completion_date and start_date > completion_date:
        errors.append("Start date must not be after the claimed completion date.")

    project_name = str(form.get("project_name") or "").strip()
    if not project_name:
        errors.append("Project name is required.")
    activities = [
        line.strip() for line in re.split(r"[\r\n]+", str(form.get("claims") or "")) if line.strip()
    ]
    if not activities:
        errors.append("At least one construction claim is required.")
    if errors:
        return errors

    previous_location = case.get("construction_location") or {}
    site_changed = (
        previous_location.get("latitude") != latitude
        or previous_location.get("longitude") != longitude
        or case.get("start_date") != start_date
        or case.get("completion_date") != completion_date
    )
    existing = {c.get("activity"): c for c in case.get("construction_claims") or []}
    claims = [existing.get(a) or {"activity": a, "claimed_status": "claimed"} for a in activities]
    claims_changed = claims != (case.get("construction_claims") or [])

    details = {
        "project_name": project_name,
        "contractor": str(form.get("contractor") or "").strip() or None,
        "construction_type": str(form.get("construction_type") or "").strip() or None,
        "payment_amount": str(form.get("payment_amount") or "").strip() or None,
    }
    details_changed = any(case.get(key) != value for key, value in details.items())

    if site_changed:
        _clear(case, _SCENE_KEYS, _RASTER_KEYS, _VISION_KEYS, _REPORT_KEYS)
    elif claims_changed:
        _clear(case, _VISION_KEYS, _REPORT_KEYS)
    elif details_changed:
        _clear(case, _REPORT_KEYS)

    bbox = [
        round(longitude - BBOX_HALF_WIDTH_DEG, 6),
        round(latitude - BBOX_HALF_WIDTH_DEG, 6),
        round(longitude + BBOX_HALF_WIDTH_DEG, 6),
        round(latitude + BBOX_HALF_WIDTH_DEG, 6),
    ]
    case.update(details)
    case.update(
        start_date=start_date,
        completion_date=completion_date,
        construction_claims=claims,
        bounding_box=bbox,
        construction_location={
            "site_name": project_name,
            "latitude": latitude,
            "longitude": longitude,
            "bbox": bbox,
        },
        review_confirmed=True,
    )
    return []


def _output_file(subdir: str, case_id: str, suffix: str) -> Path:
    directory = Path(settings.CONSTRUCTION_OUTPUT_DIR) / subdir
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{case_id}{suffix}"


def _report_bytes(report: dict[str, Any]) -> bytes:
    return json.dumps(report, indent=2, sort_keys=True, default=str).encode("utf-8")


def _save_report(case: dict[str, Any]) -> None:
    if case.get("report"):
        _output_file("reports", case["case_id"], "_report.json").write_bytes(
            _report_bytes(case["report"])
        )


def record_decision(
    case: dict[str, Any], decision: str, officer_id: str, justification: str
) -> dict[str, Any]:
    """Persist the human decision apart from the AI evidence, bound to the report's hash."""
    report = case.get("report")
    if not report:
        raise DecisionError("A verification report is required before recording a decision.")
    if case.get("officer_decision"):
        raise DecisionError("A decision has already been recorded for this case.")
    officer_id = (officer_id or "").strip()
    justification = (justification or "").strip()
    if not officer_id:
        raise DecisionError("Officer identifier is required.")
    if not justification:
        raise DecisionError("Justification is required.")

    record = OfficerDecisionRecord(
        case_id=case["case_id"],
        decision=OfficerDecisionType(decision).value,
        officer_name_or_id=officer_id,
        justification=justification,
        report_sha256=hashlib.sha256(_report_bytes(report)).hexdigest(),
        evidence_mode=report.get("evidence_mode"),
    ).model_dump()

    path = _output_file("decisions", case["case_id"], "_decision.json")
    # Exclusive mode: an audit record must never be silently overwritten.
    with open(path, "x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    case["officer_decision"] = record
    return record
