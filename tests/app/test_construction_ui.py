import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from agents.construction_verification import case_workflow as cw
from agents.construction_verification.schemas import (
    ConstructionClaim,
    ExtractedProjectInfo,
    LocationInfo,
)
from agents.construction_verification.services.earth_search import EarthSearchClient
from agents.construction_verification.services.gemini_client import GeminiVerificationClient
from agents.construction_verification.services.ocr_client import OCRClient
from agents.construction_verification.services.raster_processing import RasterProcessor
import pytest
from unittest.mock import MagicMock
import streamlit as st

from construction_ui import (
    check_missing_fields,
    init_construction_state,
    load_sample_demo_case,
    parse_coordinates_from_text,
    parse_dates_from_text,
    synthesize_recommended_action,
)
from agents.construction_verification.schemas import OfficerDecisionType, VerificationStatus


def test_parse_coordinates():
    lat, lon = parse_coordinates_from_text("Latitude: 12.9716, Longitude: 77.5946")
    assert lat == 12.9716
    assert lon == 77.5946

    lat, lon = parse_coordinates_from_text("Coordinates are 22.7215, 88.4812 in sector 4")
    assert lat == 22.7215
    assert lon == 88.4812

    lat, lon = parse_coordinates_from_text("No coordinates in this sentence")
    assert lat is None
    assert lon is None


def test_parse_dates():
    dates = parse_dates_from_text("Project from 2024-01-10 to 2024-04-15 completed")
    assert dates == ["2024-01-10", "2024-04-15"]

    dates = parse_dates_from_text("Started 2024/02/01")
    assert dates == ["2024-02-01"]


def test_missing_fields_detection():
    # Setup state
    init_construction_state()
    st.session_state.cv_lat = None
    st.session_state.cv_lon = None
    st.session_state.cv_start_date = ""
    st.session_state.cv_completion_date = ""

    missing = check_missing_fields()
    assert "Site Coordinates (Latitude & Longitude)" in missing
    assert "Project Start Date" in missing
    assert "Claimed Completion Date" in missing

    st.session_state.cv_lat = 12.9716
    st.session_state.cv_lon = 77.5946
    st.session_state.cv_start_date = "2024-01-10"
    st.session_state.cv_completion_date = "2024-04-15"

    assert len(check_missing_fields()) == 0


def test_synthesize_recommended_action_all_supported():
    init_construction_state()
    st.session_state.cv_verification_results = [
        {"claim": "Clearing", "status": VerificationStatus.SUPPORTED.value},
        {"claim": "Grading", "status": VerificationStatus.SUPPORTED.value},
    ]
    rec = synthesize_recommended_action()
    assert rec["action"] == OfficerDecisionType.APPROVE.value
    assert "APPROVE" in rec["title"]
    assert rec["badge_color"] == "green"


def test_synthesize_recommended_action_insufficient_evidence():
    init_construction_state()
    st.session_state.cv_verification_results = [
        {"claim": "Clearing", "status": VerificationStatus.SUPPORTED.value},
        {"claim": "Culvert", "status": VerificationStatus.INSUFFICIENT_EVIDENCE.value},
    ]
    rec = synthesize_recommended_action()
    assert rec["action"] == OfficerDecisionType.REQUEST_INSPECTION.value
    assert "DRONE" in rec["title"] or "INSPECTION" in rec["title"]
    assert rec["badge_color"] == "orange"


def test_synthesize_recommended_action_not_supported():
    init_construction_state()
    st.session_state.cv_verification_results = [
        {"claim": "Clearing", "status": VerificationStatus.NOT_SUPPORTED.value},
    ]
    rec = synthesize_recommended_action()
    assert rec["action"] == OfficerDecisionType.REJECT.value
    assert "REJECT" in rec["title"]
    assert rec["badge_color"] == "red"


def test_load_sample_demo_case():
    init_construction_state()
    load_sample_demo_case()
    assert st.session_state.cv_case_id == "DEMO-CASE-NH44"
    assert st.session_state.cv_lat == 12.9716
    assert st.session_state.cv_lon == 77.5946
    assert st.session_state.cv_start_date == "2024-01-10"
    assert st.session_state.cv_completion_date == "2024-04-15"
    assert len(st.session_state.cv_claims) == 3
    assert st.session_state.cv_is_demo is True


TESTS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = TESTS_DIR.parent / "src"
APP_PATH = SRC_DIR / "construction_app.py"


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _app(tmp_path, fake_model=True):
    return AppTest.from_file(str(APP_PATH), default_timeout=60), patch.multiple(
        "core.settings.settings", USE_FAKE_MODEL=fake_model, CONSTRUCTION_OUTPUT_DIR=str(tmp_path)
    )


def _seeded_case(tmp_path):
    """A case that finished the workflow in test mode, with synthetic data labelled as such."""
    case = cw.new_case()
    cw.attach_document(case, "claim.png", _png())
    case["extracted_text"] = "Fixture document text"
    assert (
        cw.apply_officer_review(
            case,
            {
                "project_name": "Fixture Road",
                "latitude": "13.05",
                "longitude": "80.25",
                "start_date": "2024-01-10",
                "completion_date": "2024-06-20",
                "claims": "Road surfacing",
            },
        )
        == []
    )
    assert cw.run_stage(case, "retrieve") == []
    cw.run_stage(case, "analyze")
    return case


def _labels(elements):
    return [e.label for e in elements]


def test_app_opens_on_five_stage_dashboard(tmp_path):
    at, settings_patch = _app(tmp_path)
    with settings_patch:
        at.run()
    assert not at.exception
    assert at.title[0].value == "Construction Claim Verification"
    assert [t.label for t in at.tabs] == [
        "1. Upload",
        "2. Review details",
        "3. Satellite images",
        "4. Analyze claims",
        "5. Report and decision",
    ]
    assert "Start New Case" in _labels(at.sidebar.button)
    assert any("TEST MODE" in e.value for e in at.error)
    assert [h.value for h in at.header] == [
        "Upload document",
        "Review extracted details",
        "Retrieve satellite images",
        "Analyze construction claims",
        "Report and human decision",
    ]
    # No chat, agent selector or discovery control is present.
    assert len(at.chat_input) == 0
    assert not any("discover" in b.label.lower() for b in at.button)


def test_app_production_mode_shows_no_test_banner(tmp_path):
    at, settings_patch = _app(tmp_path, fake_model=False)
    with settings_patch:
        at.run()
    assert not at.exception
    assert not any("TEST MODE" in e.value for e in at.error)


def test_review_form_renders_and_edits_update_case_state(tmp_path):
    at, settings_patch = _app(tmp_path)
    with settings_patch:
        case = cw.new_case()
        cw.attach_document(case, "claim.png", _png())
        case.update(extracted_text="text", extraction_ran=True)
        at.session_state["case"] = case
        at.run()
        assert not at.exception
        assert {
            "Project name", "Contractor", "Construction type", "Payment amount (if stated)",
            "Latitude (decimal degrees)", "Longitude (decimal degrees)",
            "Start date (YYYY-MM-DD)", "Claimed completion date (YYYY-MM-DD)",
        } <= set(_labels(at.text_input))
        assert any("Missing: Latitude" in w.value for w in at.warning)

        def field(label):
            return next(i for i in at.text_input if i.label == label)

        field("Project name").set_value("Edited Road")
        field("Latitude (decimal degrees)").set_value("22.5")
        field("Longitude (decimal degrees)").set_value("88.3")
        field("Start date (YYYY-MM-DD)").set_value("2023-11-05")
        field("Claimed completion date (YYYY-MM-DD)").set_value("2024-02-01")
        next(a for a in at.text_area if a.label.startswith("Construction claims")).set_value("Paving")
        next(b for b in at.button if b.label == "Confirm details").click()
        at.run()

    assert not at.exception
    state = at.session_state["case"]
    assert state["review_confirmed"] is True
    assert state["project_name"] == "Edited Road"
    assert state["construction_location"]["latitude"] == 22.5
    assert state["bounding_box"] == [88.295, 22.495, 88.305, 22.505]
    assert state["start_date"] == "2023-11-05"
    assert any(b.label == "Find Sentinel-2 scenes" for b in at.button)


def test_invalid_review_shows_errors_and_keeps_state_clean(tmp_path):
    at, settings_patch = _app(tmp_path)
    with settings_patch:
        case = cw.new_case()
        cw.attach_document(case, "claim.png", _png())
        case.update(extracted_text="text", extraction_ran=True)
        at.session_state["case"] = case
        at.run()
        next(i for i in at.text_input if i.label.startswith("Latitude")).set_value("500")
        next(b for b in at.button if b.label == "Confirm details").click()
        at.run()
    assert any("Latitude must be between" in e.value for e in at.error)
    assert not at.session_state["case"].get("review_confirmed")


def test_analysis_and_report_stages_render_labelled_test_data(tmp_path):
    at, settings_patch = _app(tmp_path)
    with settings_patch:
        at.session_state["case"] = _seeded_case(tmp_path)
        at.run()
    assert not at.exception
    assert any("SYNTHETIC TEST DATA" in e.value for e in at.error)
    assert "NDVI change" in _labels(at.metric)
    subheaders = [s.value for s in at.subheader]
    for section in ("Project", "Satellite provenance", "Spectral change", "Site progress",
                    "AI findings by claim", "Evidence limitations", "Officer decision"):
        assert section in subheaders
    assert any(e.label.startswith("VISION_NOT_EXECUTED") for e in at.expander)


def test_officer_decision_controls_require_inputs_and_record(tmp_path):
    at, settings_patch = _app(tmp_path)
    with settings_patch:
        case = _seeded_case(tmp_path)
        at.session_state["case"] = case
        at.run()
        decision = next(s for s in at.selectbox if s.label == "Decision")
        assert list(decision.options) == ["APPROVE", "REJECT", "REQUEST_INSPECTION", "HOLD"]
        assert decision.value is None

        next(b for b in at.button if b.label == "Record decision").click()
        at.run()
        assert any("Select a decision" in e.value for e in at.error)
        assert not at.session_state["case"].get("officer_decision")

        next(s for s in at.selectbox if s.label == "Decision").select("HOLD")
        next(b for b in at.button if b.label == "Record decision").click()
        at.run()
        assert any("Officer identifier is required" in e.value for e in at.error)

        next(s for s in at.selectbox if s.label == "Decision").select("HOLD")
        next(i for i in at.text_input if i.label == "Officer identifier").set_value("OFF-9")
        next(a for a in at.text_area if a.label == "Justification").set_value("Awaiting evidence")
        next(b for b in at.button if b.label == "Record decision").click()
        at.run()

    assert not at.exception
    record = at.session_state["case"]["officer_decision"]
    assert record["decision"] == "HOLD" and record["officer_name_or_id"] == "OFF-9"
    saved = tmp_path / "decisions" / f"{record['case_id']}_decision.json"
    assert json.loads(saved.read_text())["justification"] == "Awaiting evidence"
    assert not any(b.label == "Record decision" for b in at.button)


def test_start_new_case_resets_state_and_widgets(tmp_path):
    at, settings_patch = _app(tmp_path)
    with settings_patch:
        first = _seeded_case(tmp_path)
        at.session_state["case"] = first
        at.run()
        next(b for b in at.sidebar.button if b.label == "Start New Case").click()
        at.run()
    assert not at.exception
    second = at.session_state["case"]
    assert second["case_id"] != first["case_id"]
    assert set(second) == {"case_id", "document_paths", "form_revision", "stage_errors"}
    assert not at.metric and not at.expander
    assert any("Upload a PDF" in i.value for i in at.info)
    assert not any(i.label == "Project name" for i in at.text_input)


def test_reruns_do_not_repeat_external_work(tmp_path):
    at, settings_patch = _app(tmp_path)
    with (
        settings_patch,
        patch("agents.construction_verification.case_workflow.run_stage", wraps=cw.run_stage) as run,
    ):
        at.session_state["case"] = _seeded_case(tmp_path)
        before = run.call_count
        at.run()
        at.run()
        at.run()
    assert run.call_count == before


def test_app_import_excludes_unrelated_components(tmp_path):
    code = textwrap.dedent(
        f"""
        import sys
        sys.path[:0] = [{str(TESTS_DIR)!r}, {str(SRC_DIR)!r}]
        import network_guard
        network_guard.sanitize_environment()
        network_guard.install()
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file({str(APP_PATH)!r}, default_timeout=60).run()
        assert not at.exception, at.exception
        banned = [
            "agents.agents", "agents.research_assistant", "agents.langgraph_supervisor_agent",
            "agents.construction_verification.discovery_graph",
            "agents.construction_verification.services.discovery_service",
            "voice", "client",
            "service", "construction_ui", "streamlit_app",
        ]
        loaded = [name for name in banned if name in sys.modules]
        assert not loaded, loaded
        assert not network_guard.session_log, network_guard.session_log
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CONSTRUCTION_OUTPUT_DIR": str(tmp_path), "USE_FAKE_MODEL": "true"},
        cwd=SRC_DIR.parent,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "OK" in result.stdout


def test_upload_widget_renders(tmp_path):
    at, settings_patch = _app(tmp_path)
    with settings_patch:
        at.run()
    assert len(at.get("file_uploader")) == 1


def test_five_stages_driven_through_the_ui_with_mocked_services(tmp_path):
    def click(label):
        next(b for b in at.button if b.label == label).click()
        at.run()
        assert not at.exception, at.exception

    def field(label):
        return next(i for i in at.text_input if i.label == label)

    extracted = ExtractedProjectInfo(
        project_name="Ring Road",
        contractor="Acme Ltd",
        location=LocationInfo(latitude=13.05, longitude=80.25),
        start_date="2024-01-10",
        completion_date="2024-06-20",
        claims=[ConstructionClaim(activity="Road surfacing")],
    )

    def scene(scene_id, date):
        bands = ("red", "green", "blue", "nir", "scl", "swir16")
        return {
            "scene_id": scene_id,
            "acquisition_date": date,
            "cloud_cover_percent": 3.0,
            "asset_urls": {b: f"https://stac-fixture.invalid/{scene_id}/{b}.tif" for b in bands},
        }

    def read_band(self, url, bbox, target_shape=None, grid=None, resampling=None):
        before = "BEFORE" in url
        band = url.rsplit("/", 1)[-1].split(".")[0]
        value = {
            "nir": 3000.0 if before else 1500.0,
            "red": 1000.0 if before else 1500.0,
            "scl": 4,
        }.get(band, 700.0)
        return np.full(grid.shape, value, dtype=np.float32), grid.transform

    vision = {
        "findings": [
            {
                "claim": "Road surfacing",
                "status": "SUPPORTED",
                "visual_observations": ["Fixture corridor"],
                "reasoning": "Fixture reasoning",
                "limitations": ["Fixture limitation"],
                "recommended_follow_up": "Fixture follow-up",
            }
        ],
        "general_observations": "Fixture observation",
        "visible_progress": "VISIBLE_PROGRESS",
        "progress_statement": "Fixture corridor appeared.",
    }

    at, settings_patch = _app(tmp_path, fake_model=False)
    with (
        settings_patch,
        patch.object(OCRClient, "_call_ocr_space", AsyncMock(return_value="Fixture OCR text")) as ocr,
        patch.object(
            GeminiVerificationClient,
            "_ainvoke_with_fallback",
            AsyncMock(side_effect=[(extracted, "m"), (vision, "m")]),
        ) as gemini,
        patch.object(
            EarthSearchClient,
            "find_best_scene",
            side_effect=[scene("S2_BEFORE", "2024-01-05"), scene("S2_AFTER", "2024-06-25")],
        ) as stac,
        patch.object(RasterProcessor, "read_window_band", read_band),
    ):
        case = cw.new_case()
        cw.attach_document(case, "claim.png", _png())
        at.session_state["case"] = case
        at.run()

        click("Extract text")
        assert at.session_state["case"]["extracted_text"] == "Fixture OCR text"

        click("Extract details with AI")
        assert field("Project name").value == "Ring Road"
        assert field("Latitude (decimal degrees)").value == "13.05"

        field("Latitude (decimal degrees)").set_value("13.06")
        click("Confirm details")
        assert at.session_state["case"]["construction_location"]["latitude"] == 13.06

        click("Find Sentinel-2 scenes")
        assert stac.call_args_list[0].kwargs["bbox"] == [80.245, 13.055, 80.255, 13.065]

        click("Analyze claims")
        assert at.session_state["case"]["verification_results"][0]["status"] == "SUPPORTED"
        assert any(e.label == "SUPPORTED: Road surfacing" for e in at.expander)
        assert any("Visible progress on site" in s.value for s in at.success)
        assert not any("SYNTHETIC" in e.value for e in at.error)

        next(s for s in at.selectbox if s.label == "Decision").select("APPROVE")
        field("Officer identifier").set_value("OFF-1")
        next(a for a in at.text_area if a.label == "Justification").set_value("Reviewed evidence")
        click("Record decision")

        at.run()
        at.run()

    assert ocr.await_count == 1 and gemini.await_count == 2 and stac.call_count == 2
    state = at.session_state["case"]
    assert state["officer_decision"]["decision"] == "APPROVE"
    assert state["report"]["officer_decision"] is None
    assert state["report"]["evidence_mode"] == "LIVE_SERVICES"
