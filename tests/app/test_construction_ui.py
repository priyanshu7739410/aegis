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

