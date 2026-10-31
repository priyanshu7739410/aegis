import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from client import AgentClient, AgentClientError
from agents.construction_verification.schemas import (
    ConstructionClaim,
    DiscoveryCandidate,
    DiscoveryReport,
    ExtractedProjectInfo,
    LocationInfo,
    OfficerDecisionType,
    VerificationStatus,
    VisionExecutionAudit,
)
from agents.construction_verification.services.discovery_service import (
    ConstructionDiscoveryService,
    DiscoveryError,
)
from agents.construction_verification.services.earth_search import EarthSearchClient
from agents.construction_verification.services.gemini_client import GeminiVerificationClient
from agents.construction_verification.services.raster_processing import RasterProcessor
from schema import ChatMessage

logger = logging.getLogger(__name__)


def init_construction_state():
    """Initialize clean session state keys for construction verification and discovery workflows."""
    if "cv_operational_mode" not in st.session_state:
        st.session_state.cv_operational_mode = "Mode A: Verify Contractor Claim"

    # Mode A (Claim Verification) state
    if "cv_case_id" not in st.session_state:
        st.session_state.cv_case_id = f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d')}-001"
    if "cv_doc_path" not in st.session_state:
        st.session_state.cv_doc_path = None
    if "cv_doc_name" not in st.session_state:
        st.session_state.cv_doc_name = None
    if "cv_ocr_status" not in st.session_state:
        st.session_state.cv_ocr_status = "idle"
    if "cv_ocr_error" not in st.session_state:
        st.session_state.cv_ocr_error = None
    if "cv_ocr_text" not in st.session_state:
        st.session_state.cv_ocr_text = ""
    if "cv_project_name" not in st.session_state:
        st.session_state.cv_project_name = ""
    if "cv_construction_type" not in st.session_state:
        st.session_state.cv_construction_type = ""
    if "cv_lat" not in st.session_state:
        st.session_state.cv_lat = None
    if "cv_lon" not in st.session_state:
        st.session_state.cv_lon = None
    if "cv_start_date" not in st.session_state:
        st.session_state.cv_start_date = ""
    if "cv_completion_date" not in st.session_state:
        st.session_state.cv_completion_date = ""
    if "cv_claims" not in st.session_state:
        st.session_state.cv_claims = []
    if "cv_manual_corrections" not in st.session_state:
        st.session_state.cv_manual_corrections = {}
    if "cv_satellite_metrics" not in st.session_state:
        st.session_state.cv_satellite_metrics = None
    if "cv_before_scene" not in st.session_state:
        st.session_state.cv_before_scene = None
    if "cv_after_scene" not in st.session_state:
        st.session_state.cv_after_scene = None
    if "cv_verification_results" not in st.session_state:
        st.session_state.cv_verification_results = None
    if "cv_drone_images" not in st.session_state:
        st.session_state.cv_drone_images = []
    if "cv_decision_record" not in st.session_state:
        st.session_state.cv_decision_record = None
    if "cv_is_demo" not in st.session_state:
        st.session_state.cv_is_demo = False
    if "cv_last_error" not in st.session_state:
        st.session_state.cv_last_error = None
    if "cv_current_node" not in st.session_state:
        st.session_state.cv_current_node = "idle"
    if "cv_asked_for_missing_case" not in st.session_state:
        st.session_state.cv_asked_for_missing_case = None

    # Mode B (Independent Construction Discovery) state
    if "discovery_report" not in st.session_state:
        st.session_state.discovery_report = None
    if "discovery_lat" not in st.session_state:
        st.session_state.discovery_lat = 28.17
    if "discovery_lon" not in st.session_state:
        st.session_state.discovery_lon = 77.61
    if "discovery_radius_km" not in st.session_state:
        st.session_state.discovery_radius_km = 5.0
    if "discovery_before_start" not in st.session_state:
        st.session_state.discovery_before_start = "2021-01-01"
    if "discovery_before_end" not in st.session_state:
        st.session_state.discovery_before_end = "2021-03-31"
    if "discovery_after_start" not in st.session_state:
        st.session_state.discovery_after_start = "2025-01-01"
    if "discovery_after_end" not in st.session_state:
        st.session_state.discovery_after_end = "2025-03-31"
    if "discovery_min_area" not in st.session_state:
        st.session_state.discovery_min_area = 5000.0


def load_sample_demo_case():
    """Explicitly load sample NH-44 highway claim demonstration fixture."""
    st.session_state.cv_case_id = "DEMO-CASE-NH44"
    st.session_state.cv_doc_path = "samples/nh44_highway_expansion.pdf"
    st.session_state.cv_doc_name = "nh44_highway_claim.pdf"
    st.session_state.cv_ocr_status = "demo_loaded"
    st.session_state.cv_ocr_error = None
    st.session_state.cv_ocr_text = (
        "[SYNTHETIC TEST DATA — NOT A REAL CONTRACTOR CLAIM]\n"
        "CONTRACT CLAIM & PROGRESS REPORT\n"
        "Project: NH-44 Highway Expansion & Interchange Site B\n"
        "Location: Latitude 12.9716, Longitude 77.5946 (Bangalore Rural Sector)\n"
        "Construction Type: Highway Road Construction\n"
        "Start Date: 2024-01-10\n"
        "Claimed Completion Date: 2024-04-15\n"
        "Claim 1: Site clearance and complete vegetation clearing (12,000 m2) - completed.\n"
        "Claim 2: Ground grading, subgrade soil compaction, and embankment leveling - completed.\n"
        "Claim 3: Sub-base aggregate laying and reinforced concrete culvert installation - completed.\n"
        "Final payment approval requested by Contractor ABC Infra."
    )
    st.session_state.cv_project_name = "NH-44 Highway Expansion & Interchange Site B"
    st.session_state.cv_construction_type = "Highway Road Construction"
    st.session_state.cv_lat = 12.9716
    st.session_state.cv_lon = 77.5946
    st.session_state.cv_start_date = "2024-01-10"
    st.session_state.cv_completion_date = "2024-04-15"
    st.session_state.cv_claims = [
        {"activity": "Site clearance and complete vegetation clearing (12,000 m2)", "claimed_status": "completed"},
        {"activity": "Ground grading, subgrade soil compaction, and embankment leveling", "claimed_status": "completed"},
        {"activity": "Sub-base aggregate laying and reinforced concrete culvert installation", "claimed_status": "completed"},
    ]
    st.session_state.cv_is_demo = True
    st.session_state.cv_last_error = None
    st.session_state.cv_current_node = "details_extracted"


def parse_coordinates_from_text(text: str) -> tuple[float | None, float | None]:
    """Extract latitude and longitude pairs from chat message text."""
    lat, lon = None, None

    lat_match = re.search(r"(?:latitude|lat)[:\s=]+([-+]?\d{1,2}(?:\.\d+)?)", text, re.IGNORECASE)
    lon_match = re.search(r"(?:longitude|lon)[:\s=]+([-+]?\d{1,3}(?:\.\d+)?)", text, re.IGNORECASE)

    if lat_match and lon_match:
        try:
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))
        except ValueError:
            pass
    else:
        pair = re.search(r"\b([-+]?\d{1,2}\.\d{3,})[,\s]+([-+]?\d{1,3}\.\d{3,})\b", text)
        if pair:
            try:
                lat = float(pair.group(1))
                lon = float(pair.group(2))
            except ValueError:
                pass

    if lat is not None and lon is not None:
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return round(lat, 6), round(lon, 6)

    return None, None


def parse_dates_from_text(text: str) -> list[str]:
    """Extract ISO formatted dates (YYYY-MM-DD or YYYY/MM/DD) from chat message text."""
    matches = re.findall(r"\b(20\d\d[-/]\d\d[-/]\d\d)\b", text)
    return [m.replace("/", "-") for m in matches]


def check_missing_fields() -> list[str]:
    """Identify which required fields are missing for contractor claim satellite verification."""
    missing = []
    if st.session_state.cv_lat is None or st.session_state.cv_lon is None:
        missing.append("Site Coordinates (Latitude & Longitude)")
    if not st.session_state.cv_start_date:
        missing.append("Project Start Date")
    if not st.session_state.cv_completion_date:
        missing.append("Claimed Completion Date")
    return missing


def synthesize_recommended_action() -> dict[str, Any]:
    """Compute an objective action recommendation based on claim findings and physical limitations."""
    results = st.session_state.cv_verification_results
    if not results:
        return {
            "action": OfficerDecisionType.HOLD.value,
            "badge_color": "gray",
            "title": "AWAITING VERIFICATION",
            "summary": "Run Gemini AI Claim Analysis in Step 4 to compute an action recommendation.",
            "default_notes": "Pending AI verification findings.",
        }

    supported_count = sum(1 for r in results if r.get("status") == VerificationStatus.SUPPORTED.value)
    insufficient_count = sum(1 for r in results if r.get("status") == VerificationStatus.INSUFFICIENT_EVIDENCE.value)
    not_supported_count = sum(1 for r in results if r.get("status") == VerificationStatus.NOT_SUPPORTED.value)

    if not_supported_count > 0:
        return {
            "action": OfficerDecisionType.REJECT.value,
            "badge_color": "red",
            "title": "REJECT CLAIM / CALL FOR AUDIT",
            "summary": (
                f"Contradictory evidence detected! {not_supported_count} claim(s) conflict with observed "
                "multi-temporal satellite data (e.g., claimed earthworks or clearance not visible). "
                "Recommend halting payment pending contractor clarification."
            ),
            "default_notes": (
                f"REJECTION RECOMMENDED: Observed satellite imagery contradicts {not_supported_count} milestone claim(s). "
                "Full contract audit required prior to any funds disbursement."
            ),
        }
    elif insufficient_count > 0:
        return {
            "action": OfficerDecisionType.REQUEST_INSPECTION.value,
            "badge_color": "orange",
            "title": "REQUEST HIGH-RESOLUTION DRONE / ON-SITE INSPECTION",
            "summary": (
                f"Sentinel-2 10m satellite imagery verified surface changes for {supported_count} milestone(s). "
                f"However, {insufficient_count} structural or sub-surface claim(s) (e.g. culverts, layer compaction) "
                "cannot be verified at 10m resolution. Request high-resolution drone inspection before final payment."
            ),
            "default_notes": (
                f"CONDITIONAL INSPECTION REQUIRED: Earthworks verified via Sentinel-2 change detection, but {insufficient_count} "
                "sub-surface/culvert claim(s) exceed 10m satellite resolution. Requesting drone photography prior to payment."
            ),
        }
    else:
        return {
            "action": OfficerDecisionType.APPROVE.value,
            "badge_color": "green",
            "title": "APPROVE PAYMENT",
            "summary": (
                f"All {supported_count} claimed milestone(s) are corroborated by observable spectral change detection "
                "and multi-temporal satellite imagery. Ready for final officer disbursement approval."
            ),
            "default_notes": (
                f"APPROVED: Multi-temporal Sentinel-2 imagery confirms ground clearance and construction progress "
                f"across all {supported_count} milestone claim(s)."
            ),
        }


async def render_construction_verification_ui(agent_client: AgentClient, voice: Any = None, enable_audio: bool = False):
    """
    Render the AI Construction Verification & Discovery dashboard.
    Supports Mode A (Contractor Claim Verification) and Mode B (Independent Construction Discovery).
    """
    init_construction_state()
    st.session_state.thread_id = st.session_state.cv_case_id

    # Initialize assistant welcome message if chat history is empty
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        st.session_state.messages = [
            ChatMessage(
                type="ai",
                content=(
                    "👋 **Hello Officer!** I'm your Construction Verification & Remote Sensing Assistant.\n\n"
                    "I support two operational modes:\n"
                    "- **Mode A (Claim Verification)**: Verify contractor claims against satellite and drone evidence.\n"
                    "- **Mode B (Construction Discovery)**: Independently identify new construction (e.g. runways, roads, terminals) at any location like Noida Airport without a document.\n\n"
                    "You can switch modes above or ask questions directly here in chat!"
                ),
            )
        ]

    # Top Header & Mode Switcher
    st.markdown("## 🏗️ Remote Sensing Construction Verification & Discovery")
    st.caption("AI-assisted Sentinel-2 remote sensing analysis. Payment decisions are strictly human-certified.")

    col_mode, col_reset = st.columns([4, 1])
    with col_mode:
        selected_mode = st.radio(
            "Select Operational Mode",
            ["Mode A: Verify Contractor Claim", "Mode B: Discover New Construction"],
            horizontal=True,
            key="cv_mode_selector",
            help="Mode A evaluates contractor reimbursement claims. Mode B independently discovers land-cover and infrastructure changes without requiring any document.",
        )
        st.session_state.cv_operational_mode = selected_mode

    with col_reset:
        st.write("")
        if st.button("🔄 Reset Case State", use_container_width=True):
            st.session_state.cv_case_id = f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{os.urandom(3).hex().upper()}"
            st.session_state.cv_doc_path = None
            st.session_state.cv_doc_name = None
            st.session_state.cv_ocr_status = "idle"
            st.session_state.cv_ocr_text = ""
            st.session_state.cv_lat = None
            st.session_state.cv_lon = None
            st.session_state.cv_claims = []
            st.session_state.cv_satellite_metrics = None
            st.session_state.cv_verification_results = None
            st.session_state.cv_decision_record = None
            st.session_state.cv_is_demo = False
            st.session_state.discovery_report = None
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # Main 2-column layout: Workflow on Left, Chat Assistant on Right
    col_workflow, col_chat = st.columns([7, 5], gap="large")

    # =========================================================================
    # LEFT COLUMN: WORKFLOW BASED ON OPERATIONAL MODE
    # =========================================================================
    with col_workflow:
        if st.session_state.cv_operational_mode == "Mode B: Discover New Construction":
            # =====================================================================
            # MODE B: INDEPENDENT CONSTRUCTION DISCOVERY (NO CONTRACTOR DOCUMENT)
            # =====================================================================
            st.markdown("### 🛰️ Independent Construction Discovery")
            st.caption("Detects land clearing, runways, roads, and structural development between two Sentinel-2 observation windows without requiring any contractor document.")

            col_preset, col_area = st.columns([3, 2])
            with col_preset:
                preset_choice = st.selectbox(
                    "Quick Location Preset",
                    ["Noida International Airport (Jewar, Uttar Pradesh) [Benchmark]", "Custom Geographic Coordinates"],
                    index=0,
                )
            with col_area:
                area_name_in = st.text_input("Area / Project Name", value="Noida International Airport (Jewar, UP)" if "Jewar" in preset_choice else "Target Area")

            # Preset autofill
            if "Jewar" in preset_choice:
                def_lat, def_lon, def_radius = 28.17, 77.61, 5.0
                def_b_start, def_b_end = "2021-01-01", "2021-03-31"
                def_a_start, def_a_end = "2025-01-01", "2025-03-31"
            else:
                def_lat, def_lon, def_radius = st.session_state.discovery_lat, st.session_state.discovery_lon, st.session_state.discovery_radius_km
                def_b_start, def_b_end = st.session_state.discovery_before_start, st.session_state.discovery_before_end
                def_a_start, def_a_end = st.session_state.discovery_after_start, st.session_state.discovery_after_end

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                in_lat = st.number_input("Centroid Latitude (-90 to 90)", value=float(def_lat), format="%.4f")
            with col_d2:
                in_lon = st.number_input("Centroid Longitude (-180 to 180)", value=float(def_lon), format="%.4f")
            with col_d3:
                in_radius = st.number_input("Search Radius (km)", min_value=1.0, max_value=20.0, value=float(def_radius), step=0.5)

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**Pre-Construction Period (Before)**")
                b_start = st.text_input("Before Start Date (YYYY-MM-DD)", value=def_b_start)
                b_end = st.text_input("Before End Date (YYYY-MM-DD)", value=def_b_end)
            with col_t2:
                st.markdown("**Post-Construction Period (After)**")
                a_start = st.text_input("After Start Date (YYYY-MM-DD)", value=def_a_start)
                a_end = st.text_input("After End Date (YYYY-MM-DD)", value=def_a_end)

            min_area_val = st.slider("Minimum Candidate Change Area (m²)", min_value=1000, max_value=50000, value=5000, step=1000, help="Filters out small noise clusters. 10,000 m² = 1 hectare.")

            with st.expander("🌱 Optional: 3-Scene Multi-Temporal Confirmation (Seasonality Filter)", expanded=False):
                st.caption("Provide a confirmation period 3–12 months after the post-construction date to verify persistence and filter seasonal crop harvesting.")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    c_start = st.text_input("Confirmation Start Date (Optional)", value="", placeholder="e.g. 2025-10-01")
                with col_c2:
                    c_end = st.text_input("Confirmation End Date (Optional)", value="", placeholder="e.g. 2025-12-31")

            if st.button("🚀 Run Independent Construction Discovery", type="primary", use_container_width=True):
                with st.spinner("Searching Sentinel-2 STAC, aligning spectral bands, vectorizing change polygons, and executing Gemini Vision..."):
                    try:
                        disc_service = ConstructionDiscoveryService()
                        report = disc_service.run_discovery(
                            latitude=in_lat,
                            longitude=in_lon,
                            radius_km=in_radius,
                            before_date_start=b_start,
                            before_date_end=b_end,
                            after_date_start=a_start,
                            after_date_end=a_end,
                            confirmation_date_start=c_start.strip() if c_start.strip() else None,
                            confirmation_date_end=c_end.strip() if c_end.strip() else None,
                            min_change_area_m2=float(min_area_val),
                            area_name=area_name_in,
                        )
                        st.session_state.discovery_report = report
                        st.success(f"✅ Construction Discovery completed for {area_name_in}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Discovery pipeline failed: {e}")

            # Display Discovery Results
            if st.session_state.discovery_report:
                rep: DiscoveryReport = st.session_state.discovery_report
                audit: VisionExecutionAudit = rep.vision_audit

                st.markdown("---")
                st.markdown(f"### 📊 Discovery Findings: {rep.area_name}")
                st.caption(f"Discovery ID: `{rep.discovery_id}` | Centroid: `({rep.center['latitude']}, {rep.center['longitude']})` | Radius: `{rep.radius_km} km`")

                # Metrics summary
                km1, km2, km3, km4 = st.columns(4)
                km1.metric(
                    "Raw Detected Change",
                    f"{rep.raw_change_area_hectares:,.1f} ha",
                    f"{rep.raw_change_area_m2:,.0f} m² spectral change",
                )
                if rep.persistent_change_area_hectares is not None:
                    km2.metric(
                        "Persistent Change Area",
                        f"{rep.persistent_change_area_hectares:,.1f} ha",
                        f"Seasonal: {round(rep.seasonal_vegetation_area_m2 / 10000.0, 1) if rep.seasonal_vegetation_area_m2 else 0.0} ha",
                    )
                else:
                    km2.metric(
                        "Persistent Change Area",
                        "N/A (2-Scene)",
                        "Requires 3-scene confirmation",
                    )
                km3.metric("Candidate Zones", f"{rep.candidates_count} clusters", ">= min mapping area")
                km4.metric("Native Resolution", f"{rep.resolution_m:.0f} m/pixel", "Sentinel-2 L2A")

                # Honest Area Breakdown for Auditors
                st.markdown("##### 📐 Surface Area Breakdown & Classification:")
                b_c1, b_c2, b_c3, b_c4 = st.columns(4)
                b_c1.info(f"**Raw Spectral Change**\n\n**{rep.raw_change_area_hectares:,.1f} ha**\n\n`{rep.raw_change_area_m2:,.0f} m²`")
                b_c2.success(f"**Classified Construction**\n\n**{rep.classified_construction_area_hectares:,.1f} ha**\n\nIdentified works")
                b_c3.warning(f"**Agricultural / Other**\n\n**{rep.classified_agricultural_area_hectares + rep.classified_other_area_hectares:,.1f} ha**\n\nSeasonal/unclear")
                if rep.persistent_change_area_hectares is not None:
                    b_c4.success(f"**Persistent Change**\n\n**{rep.persistent_change_area_hectares:,.1f} ha**\n\nConfirmed multi-temporal")
                else:
                    b_c4.caption("**Persistent Change**\n\n*Requires 3rd confirmation date*")

                # Seasonality Advisory Banner
                if rep.seasonality_note:
                    if "2-scene analysis" in rep.seasonality_note:
                        st.warning(f"⚠️ **Seasonality Advisory**: {rep.seasonality_note}")
                    else:
                        st.success(f"🌱 **Multi-Temporal Persistence**: {rep.seasonality_note}")

                # Proof of Real Vision Execution Audit Banner
                if audit.vision_actually_executed:
                    st.success(
                        f"🟢 **Multimodal Vision Verified**: Gemini model `{audit.gemini_model_id}` processed real pixel data.\n\n"
                        f"- Image Dimensions: `{audit.rgb_image_dimensions}` | Total Image Bytes: `{sum(audit.image_byte_lengths):,} bytes`\n"
                        f"- Both Images Included: `True` | API Status: `{audit.api_call_status}`"
                    )
                else:
                    st.warning(f"⚠️ Vision status: `{audit.parsed_response_status}` — {audit.error_message or 'No image bytes analyzed'}")

                # Multi-Temporal Imagery Comparison
                st.markdown("#### Multi-Temporal Satellite Evidence:")
                ic1, ic2, ic3 = st.columns(3)
                with ic1:
                    st.caption(f"Before: {rep.before_acquisition_date} (Sentinel-2 RGB)")
                    if rep.rgb_before_path and Path(rep.rgb_before_path).exists():
                        st.image(rep.rgb_before_path, use_container_width=True)
                with ic2:
                    st.caption(f"After: {rep.after_acquisition_date} (Sentinel-2 RGB)")
                    if rep.rgb_after_path and Path(rep.rgb_after_path).exists():
                        st.image(rep.rgb_after_path, use_container_width=True)
                with ic3:
                    st.caption("Change Mask (Yellow: Disturbed Ground / Clearing)")
                    if rep.change_mask_path and Path(rep.change_mask_path).exists():
                        st.image(rep.change_mask_path, use_container_width=True)

                # Gemini Vision Analysis Interpretation
                st.markdown("#### 🤖 Gemini Vision Analysis & Structural Assessment:")
                st.info(rep.gemini_vision_summary)

                # Georeferenced Candidate Construction Features
                st.markdown(f"#### 📍 Georeferenced Candidate Construction Polygons ({len(rep.candidates)}):")
                st.caption("Coordinates and bounding boxes are derived directly from the georeferenced satellite raster grid.")

                for c in rep.candidates[:10]:
                    cls_val = c.classification.value if hasattr(c.classification, "value") else str(c.classification or "UNCLEAR")
                    badge_icons = {
                        "CONSTRUCTION": "🏗️",
                        "AGRICULTURAL_HARVEST": "🌾",
                        "WATER_BODY": "💧",
                        "CLOUD_OR_ARTIFACT": "☁️",
                        "UNCLEAR": "❓",
                    }
                    icon = badge_icons.get(cls_val, "📌")
                    conf_str = f" ({c.confidence:.0%})" if c.confidence is not None else ""
                    expander_title = f"{icon} **{c.candidate_id}** [{cls_val}{conf_str}]: {c.area_hectares} ha ({c.area_m2:,.0f} m²) — Centroid: `({c.centroid['latitude']}, {c.centroid['longitude']})`"
                    
                    with st.expander(expander_title, expanded=False):
                        if cls_val == "AGRICULTURAL_HARVEST":
                            st.warning(f"🌾 **Classification**: `{cls_val}` — Seasonal Agricultural Harvest / Clearing")
                            st.caption("ℹ️ Detected as spectral disturbance but identified as seasonal crop rotation/fallow field rather than infrastructure.")
                        elif cls_val == "CONSTRUCTION":
                            st.success(f"🏗️ **Classification**: `{cls_val}` — Infrastructure / Construction Works")
                        else:
                            st.info(f"{icon} **Classification**: `{cls_val}`")

                        if c.confidence is not None:
                            st.markdown(f"**Confidence**: `{c.confidence:.1%}`")
                        st.markdown(f"**Bounding Box**: `{c.bbox}`")
                        st.markdown(f"**Spectral Change**: Mean Δ NDVI: `{c.mean_delta_ndvi:+.3f}` | BSI Delta: `{c.mean_delta_bsi:+.3f}`")
                        if c.visual_evidence:
                            st.markdown(f"**Visual Evidence**: {c.visual_evidence}")
                        if c.limitations:
                            st.markdown(f"**Spatial Limitations (10m resolution)**: *{c.limitations}*")
                        st.markdown(f"**Interpretation**: {c.gemini_interpretation}")

                # Download Options for Google Earth Engine / GIS
                st.markdown("#### 📥 Export Data for Independent Verification:")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    geojson_str = json.dumps(rep.geojson_feature_collection, indent=2)
                    st.download_button(
                        "📥 Download GeoJSON (Google Earth Engine / QGIS)",
                        data=geojson_str,
                        file_name=f"candidates_{rep.discovery_id}.geojson",
                        mime="application/geo+json",
                        use_container_width=True,
                    )
                with dl_col2:
                    json_report_str = rep.model_dump_json(indent=2)
                    st.download_button(
                        "📥 Download Full Evidence Audit Report (JSON)",
                        data=json_report_str,
                        file_name=f"discovery_report_{rep.discovery_id}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                with st.expander("🛠️ Full Vision & Provenance Audit Log", expanded=False):
                    st.json({
                        "discovery_id": rep.discovery_id,
                        "before_scene_id": rep.before_scene_id,
                        "after_scene_id": rep.after_scene_id,
                        "confirmation_scene_id": rep.confirmation_scene_id,
                        "seasonality_note": rep.seasonality_note,
                        "retrieval_attempts": rep.retrieval_attempts,
                        "crs": rep.crs,
                        "raster_shape": rep.raster_shape,
                        "resolution_m": rep.resolution_m,
                        "vision_audit": audit.model_dump(),
                    })

        else:
            # =====================================================================
            # MODE A: VERIFY CONTRACTOR CLAIM (5-STEP DOCUMENT PIPELINE)
            # =====================================================================
            # Top demo / new case buttons
            h_col1, h_col2 = st.columns([1, 1])
            with h_col1:
                if st.button("🧪 Sample Demo (NH-44)", use_container_width=True):
                    load_sample_demo_case()
                    st.rerun()
            with h_col2:
                if st.button("🔄 New Reimbursement Case", use_container_width=True):
                    st.session_state.cv_case_id = f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{os.urandom(3).hex().upper()}"
                    st.session_state.cv_doc_path = None
                    st.session_state.cv_doc_name = None
                    st.session_state.cv_ocr_text = ""
                    st.session_state.cv_lat = None
                    st.session_state.cv_lon = None
                    st.session_state.cv_claims = []
                    st.session_state.cv_satellite_metrics = None
                    st.session_state.cv_verification_results = None
                    st.session_state.cv_decision_record = None
                    st.session_state.cv_is_demo = False
                    st.rerun()

            if st.session_state.cv_is_demo:
                st.warning("⚠️ **[SYNTHETIC TEST DATA — NOT A REAL CONTRACTOR CLAIM]**: NH-44 Highway Expansion test claim loaded. Uploading a document will overwrite this.")

            # Step 1: Upload File & OCR
            st.markdown("### 📄 Step 1: Upload File & OCR")
            st.caption("Upload contractor reimbursement requests, completion certificates, or invoice documents.")

            uploaded_file = st.file_uploader(
                "Select Contractor Document (PDF, PNG, JPG, JPEG, WEBP, TXT)",
                type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
                key="cv_step1_file_uploader",
            )

            if uploaded_file and st.button("📤 Upload & Process Document (OCR)", type="primary"):
                with st.spinner("Uploading document and running OCR text extraction..."):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        res = agent_client.upload_construction_file(
                            file_name=uploaded_file.name,
                            file_bytes=file_bytes,
                            run_ocr=True,
                            case_id=st.session_state.cv_case_id,
                        )
                        st.session_state.cv_doc_path = res.get("file_path")
                        st.session_state.cv_doc_name = uploaded_file.name
                        st.session_state.cv_case_id = res.get("case_id", st.session_state.cv_case_id)
                        st.session_state.cv_is_demo = False

                        extracted = res.get("extracted_text") or ""
                        if res.get("ocr_status") == "SUCCESS" and extracted:
                            st.session_state.cv_ocr_text = extracted
                            st.session_state.cv_ocr_status = "success"

                            with st.spinner("Gemini is parsing project details and claims..."):
                                try:
                                    gemini_client = GeminiVerificationClient()
                                    info = await gemini_client.extract_project_info(extracted)
                                    st.session_state.cv_project_name = info.project_name or ""
                                    st.session_state.cv_construction_type = info.construction_type or ""
                                    if info.location and info.location.latitude is not None:
                                        st.session_state.cv_lat = info.location.latitude
                                    if info.location and info.location.longitude is not None:
                                        st.session_state.cv_lon = info.location.longitude
                                    if info.start_date:
                                        st.session_state.cv_start_date = info.start_date
                                    if info.completion_date:
                                        st.session_state.cv_completion_date = info.completion_date
                                    if info.claims:
                                        st.session_state.cv_claims = [c.model_dump() for c in info.claims]
                                except Exception:
                                    pass
                            st.success(f"✅ OCR extraction successful ({len(extracted)} characters)!")
                            st.rerun()
                        else:
                            st.error(f"OCR Failed: {res.get('ocr_error', 'Empty text')}")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

            if st.session_state.cv_doc_name:
                st.caption(f"📎 **Active Document**: `{st.session_state.cv_doc_name}`")

            with st.expander("📄 View & Edit Extracted Document Text", expanded=False):
                edited_text = st.text_area("Document Text", value=st.session_state.cv_ocr_text, height=150)
                if st.button("💾 Save Text Changes"):
                    st.session_state.cv_ocr_text = edited_text
                    st.success("Document text saved.")

            st.markdown("---")

            # Step 2: Extracted Information & Missing Fields Check
            st.markdown("### 📋 Step 2: Extracted Details & Missing Fields Check")
            st.caption("Verify required parameters: Site Coordinates and Construction Dates are required for satellite search.")

            missing_fields = check_missing_fields()
            if missing_fields:
                st.warning(
                    f"⚠️ **Action Required**: The following parameters are missing:\n"
                    + "\n".join([f"- **{m}**" for m in missing_fields])
                    + "\n\nPlease fill them in below or tell the Chat Assistant on the right!"
                )
            else:
                st.success(
                    f"✅ **All Parameters Validated!**\n\n"
                    f"Centroid: `({st.session_state.cv_lat}, {st.session_state.cv_lon})` | Timeline: `{st.session_state.cv_start_date}` to `{st.session_state.cv_completion_date}`"
                )

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                pname = st.text_input("Project Name", value=st.session_state.cv_project_name)
                sdate = st.text_input("Start Date (YYYY-MM-DD)", value=st.session_state.cv_start_date)
                lat_str = st.text_input("Latitude", value=str(st.session_state.cv_lat) if st.session_state.cv_lat is not None else "")
            with col_p2:
                ctype = st.text_input("Construction Type", value=st.session_state.cv_construction_type)
                cdate = st.text_input("Completion Date (YYYY-MM-DD)", value=st.session_state.cv_completion_date)
                lon_str = st.text_input("Longitude", value=str(st.session_state.cv_lon) if st.session_state.cv_lon is not None else "")

            if st.session_state.cv_claims:
                st.markdown(f"**Contractor Claims ({len(st.session_state.cv_claims)}):**")
                for idx, c in enumerate(st.session_state.cv_claims):
                    st.caption(f"**Claim #{idx+1}**: {c.get('activity')} — Status: `{c.get('claimed_status', 'completed')}`")

            if st.button("💾 Save Details & Proceed to Step 3", type="primary"):
                p_lat, p_lon = None, None
                if lat_str.strip():
                    try:
                        p_lat = float(lat_str.strip())
                    except ValueError:
                        pass
                if lon_str.strip():
                    try:
                        p_lon = float(lon_str.strip())
                    except ValueError:
                        pass

                st.session_state.cv_project_name = pname.strip()
                st.session_state.cv_construction_type = ctype.strip()
                st.session_state.cv_start_date = sdate.strip().replace("/", "-")
                st.session_state.cv_completion_date = cdate.strip().replace("/", "-")
                st.session_state.cv_lat = p_lat
                st.session_state.cv_lon = p_lon
                st.success("Details saved!")
                st.rerun()

            st.markdown("---")

            # Step 3: Get Satellite Images
            st.markdown("### 🛰️ Step 3: Get Satellite Images (Sentinel-2)")
            st.caption("Fetches Sentinel-2 L2A observations before & after construction and computes spectral change detection.")

            if missing_fields:
                st.info(f"🔒 **Step 3 Locked**: Please provide {', '.join(missing_fields)} in Step 2 above first.")
            else:
                lat, lon = st.session_state.cv_lat, st.session_state.cv_lon
                s_d, c_d = st.session_state.cv_start_date, st.session_state.cv_completion_date
                bbox = [round(lon - 0.005, 6), round(lat - 0.005, 6), round(lon + 0.005, 6), round(lat + 0.005, 6)]

                st.markdown(f"**Centroid**: `({lat}, {lon})` | **Period**: `{s_d}` to `{c_d}`")

                if st.button("🛰️ Fetch Sentinel-2 Satellite Images & Detect Changes", type="primary", use_container_width=True):
                    with st.spinner("Searching Earth Search STAC for low-cloud scenes & computing NDVI change..."):
                        try:
                            stac = EarthSearchClient()
                            b_scene = stac.find_best_scene(bbox, s_d, prefer_before=True)
                            a_scene = stac.find_best_scene(bbox, c_d, prefer_before=False)

                            if not b_scene or not a_scene:
                                st.error("No low-cloud Sentinel-2 scenes found for target dates.")
                            else:
                                st.session_state.cv_before_scene = b_scene
                                st.session_state.cv_after_scene = a_scene

                                proc = RasterProcessor()
                                metrics = proc.analyze_change(
                                    case_id=st.session_state.cv_case_id,
                                    bbox_wgs84=bbox,
                                    before_scene=b_scene,
                                    after_scene=a_scene,
                                )
                                st.session_state.cv_satellite_metrics = metrics
                                st.success("✅ Satellite spectral change detection complete!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Satellite analysis failed: {e}")

            if st.session_state.cv_satellite_metrics:
                m = st.session_state.cv_satellite_metrics
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Vegetation Loss", f"{m['vegetation_loss_fraction']*100:.1f}%", f"{m['delta_ndvi']:.3f} Δ NDVI")
                sm2.metric("Disturbed Ground", f"{m['potential_disturbed_area_m2']:,.0f} m²")
                sm3.metric("Observation Coverage", f"{m['valid_observation_coverage_pct']:.1f}%", "Cloud-free")

                simg1, simg2, simg3 = st.columns(3)
                with simg1:
                    st.caption("Before (Sentinel-2)")
                    if m.get("rgb_before_preview_path") and Path(m["rgb_before_preview_path"]).exists():
                        st.image(m["rgb_before_preview_path"], use_container_width=True)
                with simg2:
                    st.caption("After (Sentinel-2)")
                    if m.get("rgb_after_preview_path") and Path(m["rgb_after_preview_path"]).exists():
                        st.image(m["rgb_after_preview_path"], use_container_width=True)
                with simg3:
                    st.caption("NDVI Change Colormap")
                    if m.get("ndvi_diff_image_path") and Path(m["ndvi_diff_image_path"]).exists():
                        st.image(m["ndvi_diff_image_path"], use_container_width=True)

            st.markdown("---")

            # Step 4: Gemini Claim Verification
            st.markdown("### 🤖 Step 4: Gemini AI Claim Analysis")
            st.caption("Gemini Vision analyzes multi-temporal imagery and spectral change metrics against each contractor claim.")

            if not st.session_state.cv_satellite_metrics or not st.session_state.cv_claims:
                st.info("🔒 **Step 4 Locked**: Please fetch satellite images in Step 3 above first.")
            else:
                if st.button("🤖 Run Gemini AI Claim Analysis", type="primary", use_container_width=True):
                    with st.spinner("Gemini Vision is evaluating claims against visible satellite evidence..."):
                        try:
                            client = GeminiVerificationClient()
                            m = st.session_state.cv_satellite_metrics
                            vision_out = await client.analyze_satellite_imagery(
                                claims=st.session_state.cv_claims,
                                before_image_path=m["rgb_before_preview_path"],
                                after_image_path=m["rgb_after_preview_path"],
                                metrics=m,
                                before_date=st.session_state.cv_start_date,
                                after_date=st.session_state.cv_completion_date,
                            )

                            results = []
                            for c in st.session_state.cv_claims:
                                finding = next((f for f in vision_out.findings if f.claim == c.get("activity")), None)
                                if finding and finding.supports_claim is True:
                                    st_val = VerificationStatus.SUPPORTED.value
                                    action = "Observable changes support claim. Officer review recommended."
                                elif finding and finding.supports_claim is False:
                                    st_val = VerificationStatus.NOT_SUPPORTED.value
                                    action = "Contradictory evidence. Demand explanation or reject."
                                else:
                                    st_val = VerificationStatus.INSUFFICIENT_EVIDENCE.value
                                    action = "Request high-resolution drone imagery or on-site physical inspection."

                                results.append({
                                    "claim": c.get("activity"),
                                    "status": st_val,
                                    "observations": finding.observed_changes if finding else [],
                                    "reason": finding.reasoning if finding else "Inconclusive.",
                                    "limitations": finding.limitations if finding else ["10m Sentinel-2 pixel limitation"],
                                    "recommended_next_action": action,
                                })

                            st.session_state.cv_verification_results = results
                            st.success("✅ Gemini verification complete!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Gemini analysis failed: {e}")

            if st.session_state.cv_verification_results:
                st.markdown("#### Verification Findings:")
                for res in st.session_state.cv_verification_results:
                    status = res["status"]
                    icon = "✅" if status == "SUPPORTED" else "❌" if status == "NOT_SUPPORTED" else "⚠️"
                    with st.expander(f"{icon} **{res['claim']}** — `{status}`", expanded=True):
                        st.markdown(f"**Status**: `{status}`")
                        st.markdown(f"**Gemini Reasoning**: {res['reason']}")
                        st.info(f"**Recommended Action**: {res['recommended_next_action']}")

            st.markdown("---")

            # Step 5: Suggested Action & Officer Decision
            st.markdown("### ⚖️ Step 5: Suggested Action & Official Decision")
            st.caption("AI evaluates all multimodal evidence to recommend an official action. Payment decisions are strictly human-certified.")

            if not st.session_state.cv_verification_results:
                st.info("🔒 **Step 5 Locked**: Please run Gemini Claim Verification in Step 4 above first.")
            else:
                rec = synthesize_recommended_action()
                if rec["badge_color"] == "green":
                    st.success(f"🎯 **AI SUGGESTED ACTION**: `{rec['title']}`\n\n{rec['summary']}")
                elif rec["badge_color"] == "orange":
                    st.warning(f"🎯 **AI SUGGESTED ACTION**: `{rec['title']}`\n\n{rec['summary']}")
                else:
                    st.error(f"🎯 **AI SUGGESTED ACTION**: `{rec['title']}`\n\n{rec['summary']}")

                col_dec1, col_dec2 = st.columns(2)
                with col_dec1:
                    officer_id = st.text_input("Officer ID", value="", placeholder="e.g. PWD-OFFICER-104")
                    decision_options = [
                        OfficerDecisionType.APPROVE.value,
                        OfficerDecisionType.REQUEST_INSPECTION.value,
                        OfficerDecisionType.REJECT.value,
                        OfficerDecisionType.HOLD.value,
                    ]
                    default_idx = decision_options.index(rec["action"]) if rec["action"] in decision_options else 0
                    sel_dec = st.selectbox("Official Decision", options=decision_options, index=default_idx)
                with col_dec2:
                    dec_notes = st.text_area("Findings & Justification", value=rec["default_notes"], height=100)

                if st.button("⚖️ Confirm & Record Official Decision", type="primary", use_container_width=True):
                    if not officer_id.strip():
                        st.error("Please enter an Officer ID to authorize this audit certificate.")
                    else:
                        try:
                            res = agent_client.record_officer_decision(
                                case_id=st.session_state.cv_case_id,
                                decision=sel_dec,
                                officer_name_or_id=officer_id.strip(),
                                notes=dec_notes,
                            )
                            st.session_state.cv_decision_record = res.get("decision_record")
                            st.success(f"✅ Official Decision Recorded: `{sel_dec}`!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to record decision: {e}")

                if st.session_state.cv_decision_record:
                    st.markdown("#### 📜 Official Audit Certificate")
                    st.json(st.session_state.cv_decision_record)

    # =========================================================================
    # RIGHT COLUMN: LIVE CHAT ASSISTANT
    # =========================================================================
    with col_chat:
        st.markdown("### 💬 Assistant & Live Help")

        if st.session_state.cv_operational_mode == "Mode B: Discover New Construction":
            if st.session_state.discovery_report:
                st.success("🛰️ **Discovery Active**: Discovered features ready to inspect.")
            else:
                st.info("🛰️ **Ready to Discover**: Select location and click 'Run Discovery'.")
        else:
            missing = check_missing_fields()
            if missing and st.session_state.cv_ocr_text:
                st.warning("⚠️ **Help Needed**: Missing coordinates or dates for satellite search.")
            elif not st.session_state.cv_satellite_metrics and not missing:
                st.info("🛰️ **Next Step**: Click 'Fetch Sentinel-2 Satellite Images' in Step 3.")
            elif st.session_state.cv_satellite_metrics and not st.session_state.cv_verification_results:
                st.info("🤖 **Next Step**: Click 'Run Gemini Claim Analysis' in Step 4.")
            elif st.session_state.cv_verification_results and not st.session_state.cv_decision_record:
                st.success("⚖️ **Next Step**: Review the suggested action and record decision in Step 5.")
            elif st.session_state.cv_decision_record:
                st.success("✅ **Case Complete**: Official payment decision recorded!")
            else:
                st.caption("Ready to assist. Upload document in Step 1 or ask a question.")

        chat_container = st.container(height=520)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg.type):
                    st.write(msg.content)

        # Quick action pills
        pill_col1, pill_col2, pill_col3 = st.columns(3)
        with pill_col1:
            if st.session_state.cv_operational_mode == "Mode B: Discover New Construction":
                if st.button("✈️ Jewar Airport", use_container_width=True):
                    st.session_state.discovery_lat = 28.17
                    st.session_state.discovery_lon = 77.61
                    st.session_state.messages.append(ChatMessage(type="human", content="Focus discovery on Noida International Airport, Jewar (28.17, 77.61)"))
                    st.session_state.messages.append(ChatMessage(type="ai", content="📍 Coordinates set to Jewar Airport (28.17, 77.61). Ready to run discovery!"))
                    st.rerun()
            else:
                if st.button("📍 Sample Coords", use_container_width=True):
                    st.session_state.cv_lat = 12.9716
                    st.session_state.cv_lon = 77.5946
                    if not st.session_state.cv_start_date:
                        st.session_state.cv_start_date = "2024-01-10"
                    if not st.session_state.cv_completion_date:
                        st.session_state.cv_completion_date = "2024-04-15"
                    st.session_state.messages.append(ChatMessage(type="human", content="Please set coordinates to 12.9716, 77.5946"))
                    st.session_state.messages.append(ChatMessage(type="ai", content="📍 **Coordinates Set**: Lat `12.9716`, Lon `77.5946`. Step 3 is ready!"))
                    st.rerun()

        with pill_col2:
            if st.session_state.cv_operational_mode == "Mode B: Discover New Construction":
                if st.button("🔍 Provenance", use_container_width=True):
                    st.session_state.messages.append(ChatMessage(type="human", content="Show satellite scene provenance"))
                    if st.session_state.discovery_report:
                        rep = st.session_state.discovery_report
                        resp = (
                            f"🛰️ **Scene Provenance**:\n"
                            f"- Before Scene: `{rep.before_scene_id}` on `{rep.before_acquisition_date}` (Cloud: {rep.before_cloud_cover_pct:.1f}%)\n"
                            f"- After Scene: `{rep.after_scene_id}` on `{rep.after_acquisition_date}` (Cloud: {rep.after_cloud_cover_pct:.1f}%)\n"
                            f"- CRS: `{rep.crs}` | Resolution: `{rep.resolution_m} m/px`"
                        )
                    else:
                        resp = "Run discovery first to inspect Sentinel-2 scene provenance."
                    st.session_state.messages.append(ChatMessage(type="ai", content=resp))
                    st.rerun()
            else:
                if st.button("❓ What's missing?", use_container_width=True):
                    curr_missing = check_missing_fields()
                    st.session_state.messages.append(ChatMessage(type="human", content="What fields are missing?"))
                    resp = f"Missing: {', '.join(curr_missing) if curr_missing else 'All parameters present and ready!'}"
                    st.session_state.messages.append(ChatMessage(type="ai", content=resp))
                    st.rerun()

        with pill_col3:
            if st.session_state.cv_operational_mode == "Mode B: Discover New Construction":
                if st.button("📐 Candidates", use_container_width=True):
                    st.session_state.messages.append(ChatMessage(type="human", content="List candidate construction coordinates"))
                    if st.session_state.discovery_report and st.session_state.discovery_report.candidates:
                        lines = ["📍 **Top Candidate Centroids**:"]
                        for c in st.session_state.discovery_report.candidates[:5]:
                            lines.append(f"- **{c.candidate_id}**: `({c.centroid['latitude']}, {c.centroid['longitude']})` — {c.area_hectares} ha")
                        resp = "\n".join(lines)
                    else:
                        resp = "No candidate construction polygons detected yet."
                    st.session_state.messages.append(ChatMessage(type="ai", content=resp))
                    st.rerun()
            else:
                if st.button("💡 Explain Action", use_container_width=True):
                    rec = synthesize_recommended_action()
                    st.session_state.messages.append(ChatMessage(type="human", content="Explain suggested action"))
                    st.session_state.messages.append(ChatMessage(type="ai", content=f"🎯 **{rec['title']}**: {rec['summary']}"))
                    st.rerun()

        # Chat Input
        user_prompt = st.chat_input("Ask a question, enter coordinates, or discuss findings...", key="cv_chat_input")
        if user_prompt:
            st.session_state.messages.append(ChatMessage(type="human", content=user_prompt))

            parsed_lat, parsed_lon = parse_coordinates_from_text(user_prompt)
            parsed_dates = parse_dates_from_text(user_prompt)

            if parsed_lat is not None and parsed_lon is not None:
                if st.session_state.cv_operational_mode == "Mode B: Discover New Construction":
                    st.session_state.discovery_lat = parsed_lat
                    st.session_state.discovery_lon = parsed_lon
                    response_text = f"📍 Discovery center set to `({parsed_lat}, {parsed_lon})`. Click 'Run Independent Construction Discovery' to analyze."
                else:
                    st.session_state.cv_lat = parsed_lat
                    st.session_state.cv_lon = parsed_lon
                    response_text = f"📍 Coordinates updated to `({parsed_lat}, {parsed_lon})`. Step 3 is now ready!"
            elif len(parsed_dates) >= 2 and st.session_state.cv_operational_mode == "Mode A: Verify Contractor Claim":
                st.session_state.cv_start_date = parsed_dates[0]
                st.session_state.cv_completion_date = parsed_dates[1]
                response_text = f"📅 Dates updated to `{parsed_dates[0]}` to `{parsed_dates[1]}`."
            else:
                lower = user_prompt.lower()
                if "jewar" in lower or "airport" in lower or "noida" in lower:
                    if st.session_state.discovery_report:
                        rep = st.session_state.discovery_report
                        response_text = (
                            f"✈️ **Jewar Airport Analysis**: {rep.total_disturbed_area_hectares:,.1f} hectares of transformed ground detected.\n\n"
                            f"**Gemini Vision Finding**: {rep.gemini_vision_summary[:400]}..."
                        )
                    else:
                        response_text = "To analyze Jewar Airport, switch to Mode B (Discover New Construction) above and click 'Run Independent Construction Discovery'!"
                elif "provenance" in lower or "scene" in lower:
                    if st.session_state.discovery_report:
                        rep = st.session_state.discovery_report
                        response_text = f"🛰️ Sentinel-2 Scenes: Before: `{rep.before_scene_id}` ({rep.before_acquisition_date}), After: `{rep.after_scene_id}` ({rep.after_acquisition_date})."
                    else:
                        response_text = "No discovery report active yet."
                else:
                    response_text = "I'm your Remote Sensing Assistant. In Mode A, I help verify contractor invoices. In Mode B, I discover real construction changes from satellite imagery."

            st.session_state.messages.append(ChatMessage(type="ai", content=response_text))
            st.rerun()
