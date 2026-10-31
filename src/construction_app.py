"""Construction claim verification dashboard. Run: uv run streamlit run src/construction_app.py"""

from pathlib import Path
from typing import Any

import streamlit as st

from agents.construction_verification.case_workflow import (
    DecisionError,
    apply_officer_review,
    attach_document,
    missing_review_fields,
    new_case,
    record_decision,
    review_form_values,
    run_stage,
    stage_complete,
)
from agents.construction_verification.schemas import OfficerDecisionType
from core.settings import settings


def _show_errors(case: dict[str, Any], stage: str) -> None:
    for error in case["stage_errors"].get(stage, []):
        st.error(error)


def _run(case: dict[str, Any], stage: str, label: str, force: bool = False) -> None:
    with st.spinner(label):
        run_stage(case, stage, force=force)
    st.rerun()


def _render_sidebar(case: dict[str, Any]) -> None:
    with st.sidebar:
        st.subheader("Current case")
        st.code(case["case_id"], language=None)
        if st.button("Start New Case", icon=":material/add:", width="stretch"):
            st.session_state.case = new_case()
            st.rerun()

        st.subheader("Progress")
        st.write(f"{'✅' if stage_complete(case, 'ocr') else '⬜'} 1. Document text")
        st.write(f"{'✅' if case.get('review_confirmed') else '⬜'} 2. Details confirmed")
        st.write(f"{'✅' if stage_complete(case, 'retrieve') else '⬜'} 3. Satellite scenes")
        st.write(f"{'✅' if case.get('satellite_metrics') else '⬜'} 4. Claims analyzed")
        st.write(f"{'✅' if case.get('officer_decision') else '⬜'} 5. Officer decision")
        st.caption("AI output is evidence only. Only an officer can decide on payment.")


def _render_upload(case: dict[str, Any]) -> None:
    st.header("Upload document")
    uploaded = st.file_uploader(
        "Contractor claim document",
        type=["pdf", "png", "jpg", "jpeg"],
        key=f"upload_{case['case_id']}",
    )
    if uploaded is not None:
        try:
            attach_document(case, uploaded.name, uploaded.getvalue())
        except ValueError as e:
            st.error(str(e))
            return

    if not case.get("document_paths"):
        st.info("Upload a PDF, PNG, JPG or JPEG to begin.")
        return

    path = Path(case["document_paths"][0])
    st.caption(f"Attached: {case['document_name']}")
    if path.suffix.lower() == ".pdf":
        st.download_button(
            "Download uploaded PDF", path.read_bytes(), file_name=case["document_name"]
        )
    else:
        st.image(str(path), caption=case["document_name"], width=480)

    done = stage_complete(case, "ocr")
    if st.button(
        "Re-run text extraction" if done else "Extract text",
        type="secondary" if done else "primary",
        disabled=bool(case.get("officer_decision")),
    ):
        _run(case, "ocr", "Extracting document text...", force=done)

    _show_errors(case, "ocr")
    if case.get("extracted_text"):
        st.text_area(
            "Extracted text",
            case["extracted_text"],
            height=280,
            disabled=True,
            key=f"ocr_text_{case['case_id']}_{case.get('document_sha256')}",
        )
        st.success("Text extracted. Continue to Review details.")


def _render_review(case: dict[str, Any]) -> None:
    st.header("Review extracted details")
    if not case.get("extracted_text"):
        st.info("Extract the document text in stage 1 first.")
        return

    ran = stage_complete(case, "extract")
    if st.button(
        "Re-run AI extraction (discards edits)" if ran else "Extract details with AI",
        type="secondary" if ran else "primary",
        disabled=bool(case.get("officer_decision")),
    ):
        _run(case, "extract", "Extracting project details...", force=ran)

    _show_errors(case, "extract")
    if case.get("extraction_error"):
        st.warning("Automatic extraction failed. You can enter the details by hand below.")
    elif not ran:
        return

    for field in missing_review_fields(case):
        st.warning(f"Missing: {field}")

    values = review_form_values(case)
    # Keys carry the case and extraction revision so stale widget values can never reappear.
    prefix = f"review_{case['case_id']}_{case['form_revision']}"
    with st.form(prefix):
        left, right = st.columns(2)
        project_name = left.text_input("Project name", values["project_name"])
        contractor = right.text_input("Contractor", values["contractor"])
        construction_type = left.text_input("Construction type", values["construction_type"])
        payment_amount = right.text_input("Payment amount (if stated)", values["payment_amount"])
        latitude = left.text_input("Latitude (decimal degrees)", values["latitude"])
        longitude = right.text_input("Longitude (decimal degrees)", values["longitude"])
        start_date = left.text_input("Start date (YYYY-MM-DD)", values["start_date"])
        completion_date = right.text_input(
            "Claimed completion date (YYYY-MM-DD)", values["completion_date"]
        )
        claims = st.text_area("Construction claims (one per line)", values["claims"], height=140)
        submitted = st.form_submit_button(
            "Confirm details", type="primary", disabled=bool(case.get("officer_decision"))
        )

    if submitted:
        errors = apply_officer_review(
            case,
            {
                "project_name": project_name,
                "contractor": contractor,
                "construction_type": construction_type,
                "payment_amount": payment_amount,
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "completion_date": completion_date,
                "claims": claims,
            },
        )
        for error in errors:
            st.error(error)
        if not errors:
            st.rerun()
    elif case.get("review_confirmed"):
        st.success("Details confirmed. Continue to Satellite images.")


def _render_scene(scene: dict[str, Any], title: str) -> None:
    st.subheader(title)
    st.write(f"Scene ID: `{scene.get('scene_id')}`")
    st.write(f"Acquired: {scene.get('acquisition_date')}")
    st.write(f"Scene cloud cover: {scene.get('cloud_cover_percent')}%")
    st.caption(f"Source: {settings.EARTH_SEARCH_URL} ({settings.EARTH_SEARCH_COLLECTION})")


def _render_retrieval(case: dict[str, Any]) -> None:
    st.header("Retrieve satellite images")
    if not case.get("review_confirmed"):
        st.info("Confirm the project details in stage 2 first.")
        return

    st.write(
        f"Searching around ({case['construction_location']['latitude']}, "
        f"{case['construction_location']['longitude']}) for "
        f"{case['start_date']} to {case['completion_date']}."
    )
    done = stage_complete(case, "retrieve")
    if st.button(
        "Search again" if done else "Find Sentinel-2 scenes",
        type="secondary" if done else "primary",
        disabled=bool(case.get("officer_decision")),
    ):
        _run(case, "retrieve", "Searching Sentinel-2 L2A scenes...", force=done)

    _show_errors(case, "retrieve")
    if done:
        left, right = st.columns(2)
        with left:
            _render_scene(case["before_scene"], "Before")
        with right:
            _render_scene(case["after_scene"], "After")
        st.success("Scenes found. Continue to Analyze claims.")


def _metric(value: Any, suffix: str = "") -> str:
    return "Not available" if value is None else f"{value}{suffix}"


def _render_measurements(metrics: dict[str, Any]) -> None:
    columns = st.columns(4)
    columns[0].metric("NDVI before", _metric(metrics.get("mean_ndvi_before")))
    columns[1].metric("NDVI after", _metric(metrics.get("mean_ndvi_after")))
    columns[2].metric("NDVI change", _metric(metrics.get("delta_ndvi")))
    columns[3].metric("BSI change", _metric(metrics.get("delta_bsi")))
    columns = st.columns(3)
    loss = metrics.get("vegetation_loss_fraction")
    columns[0].metric("Vegetation loss", "Not available" if loss is None else f"{loss * 100:.1f}%")
    columns[1].metric(
        "Spectrally disturbed area", _metric(metrics.get("potential_disturbed_area_m2"), " m²")
    )
    columns[2].metric(
        "Cloud-free coverage", _metric(metrics.get("valid_observation_coverage_pct"), "%")
    )


def _render_imagery(metrics: dict[str, Any], case: dict[str, Any]) -> None:
    images = (
        ("rgb_before_preview_path", f"Before ({(case.get('before_scene') or {}).get('acquisition_date')})"),
        ("rgb_after_preview_path", f"After ({(case.get('after_scene') or {}).get('acquisition_date')})"),
        ("ndvi_diff_image_path", "NDVI change (red: loss, green: gain)"),
    )
    columns = st.columns(3)
    for column, (key, caption) in zip(columns, images):
        path = metrics.get(key)
        if path and Path(path).exists():
            column.image(path, caption=caption, width="stretch")
        else:
            column.warning(f"{caption}: image unavailable")


def _render_progress(visible_progress: str | None, statement: str | None) -> None:
    """Overall site change, shown apart from claim verdicts, which can stay inconclusive."""
    if not visible_progress:
        return
    text = statement or "The model gave no description."
    if visible_progress == "VISIBLE_PROGRESS":
        st.success(f"**Visible progress on site.** {text}")
    elif visible_progress == "NO_VISIBLE_PROGRESS":
        st.error(f"**No visible progress on site.** {text}")
    else:
        st.info(f"**Site progress unclear.** {text}")


def _render_analysis(case: dict[str, Any]) -> None:
    st.header("Analyze construction claims")
    if not stage_complete(case, "retrieve"):
        st.info("Retrieve both satellite scenes in stage 3 first.")
        return

    done = stage_complete(case, "analyze")
    attempted = bool(case.get("vision_audit"))
    if st.button(
        "Re-run analysis" if done else ("Retry vision analysis" if attempted else "Analyze claims"),
        type="secondary" if done else "primary",
        disabled=bool(case.get("officer_decision")),
    ):
        _run(case, "analyze", "Computing spectral change and running vision analysis...", force=done)

    _show_errors(case, "analyze")
    if case.get("satellite_error"):
        st.error(f"Satellite analysis failed: {case['satellite_error']}")
    metrics = case.get("satellite_metrics") or {}
    if metrics:
        _render_measurements(metrics)
        _render_imagery(metrics, case)
    if case.get("verification_results"):
        audit = case.get("vision_audit") or {}
        if not audit.get("vision_actually_executed"):
            st.error(f"VISION_NOT_EXECUTED: {audit.get('error_message') or 'The vision model did not run.'}")
        report = case.get("report") or {}
        _render_progress(report.get("visible_progress"), report.get("progress_statement"))
        st.dataframe(
            [
                {"Claim": r["claim"], "Status": r["status"]}
                for r in case["verification_results"]
            ],
            hide_index=True,
        )
        st.success("Analysis finished. Continue to Report and decision.")


def _render_report_body(case: dict[str, Any], report: dict[str, Any]) -> None:
    if report.get("evidence_mode") == "SYNTHETIC_TEST_DATA":
        st.error("SYNTHETIC TEST DATA. This report contains no genuine satellite evidence.")

    st.subheader("Project")
    location = report.get("location") or {}
    st.table(
        {
            "Field": ["Project", "Contractor", "Type", "Coordinates", "Period", "Payment claimed"],
            "Value": [
                report.get("project_name"),
                report.get("contractor") or "Not stated",
                report.get("construction_type"),
                f"{location.get('latitude')}, {location.get('longitude')}",
                f"{report.get('start_date')} to {report.get('completion_date')}",
                report.get("payment_amount") or "Not stated",
            ],
        }
    )

    st.subheader("Satellite provenance")
    scenes = [("Before", report.get("before_scene")), ("After", report.get("after_scene"))]
    available = [(label, scene) for label, scene in scenes if scene]
    if available:
        st.dataframe(
            [
                {
                    "Scene": label,
                    "ID": scene.get("scene_id"),
                    "Acquired": scene.get("acquisition_date"),
                    "Cloud cover %": scene.get("cloud_cover_percent"),
                }
                for label, scene in available
            ],
            hide_index=True,
        )
    else:
        st.warning("No satellite scene contributed to this report.")

    metrics = report.get("satellite_metrics") or {}
    st.subheader("Spectral change")
    if metrics:
        _render_measurements(metrics)
        _render_imagery(metrics, case)
    else:
        st.warning("No spectral measurements are available for this report.")

    st.subheader("Site progress")
    if report.get("visible_progress"):
        _render_progress(report["visible_progress"], report.get("progress_statement"))
    else:
        st.warning("No progress assessment is available because vision analysis did not run.")

    st.subheader("AI findings by claim")
    model_used = (report.get("vision_audit") or {}).get("model_used")
    st.caption(f"Vision model: {model_used or 'none, vision did not run'}")
    for result in report.get("claims_summary") or []:
        with st.expander(f"{result['status']}: {result['claim']}", expanded=True):
            for observation in result.get("observations") or []:
                st.write(f"- {observation}")
            st.write(f"**Reasoning:** {result.get('reason')}")
            for limitation in result.get("limitations") or []:
                st.caption(f"Limitation: {limitation}")
            st.write(f"**Recommended follow-up:** {result.get('recommended_next_action')}")
    if report.get("general_observations"):
        st.write(report["general_observations"])

    st.subheader("Evidence limitations")
    for limitation in report.get("evidence_limitations") or []:
        st.warning(limitation)


def _render_decision(case: dict[str, Any]) -> None:
    st.subheader("Officer decision")
    st.caption("The AI does not authorize payment. This decision is recorded separately from the evidence.")
    record = case.get("officer_decision")
    if record:
        st.success(
            f"{record['decision']} recorded by {record['officer_name_or_id']} at {record['timestamp']}."
        )
        st.write(f"Justification: {record['justification']}")
        return

    with st.form(f"decision_{case['case_id']}"):
        decision = st.selectbox("Decision", [d.value for d in OfficerDecisionType], index=None)
        officer_id = st.text_input("Officer identifier")
        justification = st.text_area("Justification")
        submitted = st.form_submit_button("Record decision", type="primary")

    if submitted:
        if decision is None:
            st.error("Select a decision.")
            return
        try:
            record_decision(case, decision, officer_id, justification)
        except DecisionError as e:
            st.error(str(e))
            return
        st.rerun()


def _render_report(case: dict[str, Any]) -> None:
    st.header("Report and human decision")
    if not stage_complete(case, "report"):
        if case.get("satellite_metrics") or case.get("vision_audit"):
            # Officer edited project details only, and rebuilding the report calls no external service.
            run_stage(case, "report")
            _show_errors(case, "report")
        if not stage_complete(case, "report"):
            st.info("Run the analysis in stage 4 first.")
            return
    _render_report_body(case, case["report"])
    _render_decision(case)


def render_dashboard() -> None:
    st.session_state.setdefault("case", new_case())
    case = st.session_state.case

    st.title("Construction Claim Verification")
    if settings.USE_FAKE_MODEL:
        st.error(
            "TEST MODE (USE_FAKE_MODEL=true). External services are off and all results are "
            "synthetic. Do not use them as evidence."
        )

    _render_sidebar(case)
    upload, review, retrieval, analysis, report = st.tabs(
        ["1. Upload", "2. Review details", "3. Satellite images", "4. Analyze claims", "5. Report and decision"]
    )
    with upload:
        _render_upload(case)
    with review:
        _render_review(case)
    with retrieval:
        _render_retrieval(case)
    with analysis:
        _render_analysis(case)
    with report:
        _render_report(case)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Construction Claim Verification",
        page_icon=":material/satellite_alt:",
        layout="wide",
    )
    render_dashboard()
