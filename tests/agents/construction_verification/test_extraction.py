import pytest
from unittest.mock import AsyncMock, patch

from agents.construction_verification.schemas import (
    ConstructionClaim,
    ExtractedProjectInfo,
    LocationInfo,
)
from agents.construction_verification.services.gemini_client import (
    GeminiExtractionError,
    GeminiVerificationClient,
)


@pytest.mark.asyncio
async def test_construction_claim_extraction():
    """Test 3: Construction claim extraction from document text into structured models (mocked LLM)."""
    sample_ocr = """
    PROGRESS PAYMENT CLAIM
    Project: Green Valley Ring Road Section 4
    Construction Type: Road and Highway Works
    Location: Latitude 13.0827, Longitude 80.2707
    Start Date: 2024-02-01
    Claimed Completion Date: 2024-05-15
    Claim 1: Vegetation clearing and topsoil stripping (15,000 sq.m) - completed
    Claim 2: Sub-base compaction and aggregate spreading - completed
    """

    mock_extracted = ExtractedProjectInfo(
        project_name="Green Valley Ring Road Section 4",
        construction_type="Road and Highway Works",
        location=LocationInfo(site_name="Green Valley Ring Road", latitude=13.0827, longitude=80.2707, bbox=[80.2657, 13.0777, 80.2757, 13.0877]),
        start_date="2024-02-01",
        completion_date="2024-05-15",
        claims=[
            ConstructionClaim(activity="Vegetation clearing and topsoil stripping (15,000 sq.m)", claimed_status="completed", quantity="15,000 sq.m"),
            ConstructionClaim(activity="Sub-base compaction and aggregate spreading", claimed_status="completed"),
        ],
    )

    client = GeminiVerificationClient(api_key="mock_key")
    with (
        patch("core.settings.settings.USE_FAKE_MODEL", False),
        patch.object(client, "_ainvoke_with_fallback", return_value=(mock_extracted, "gemini-3.8-flash")),
    ):
        result = await client.extract_project_info(sample_ocr)

    assert isinstance(result, ExtractedProjectInfo)
    assert "Green Valley" in result.project_name or "Ring Road" in result.project_name
    assert result.location is not None
    assert pytest.approx(result.location.latitude, 0.01) == 13.0827
    assert pytest.approx(result.location.longitude, 0.01) == 80.2707
    assert result.start_date == "2024-02-01"
    assert result.completion_date == "2024-05-15"
    assert len(result.claims) == 2
    assert any("clearing" in c.activity.lower() for c in result.claims)


@pytest.mark.asyncio
async def test_missing_location_and_dates_detection():
    """Test 4: Missing location coordinates and dates are accurately flagged."""
    incomplete_ocr = """
    GENERAL INVOICE
    The contractor claims that site clearance and earth leveling works have been finished.
    Payment requested.
    """

    client = GeminiVerificationClient(api_key="mock_key")
    model_output = ExtractedProjectInfo(
        project_name="General Invoice",
        claims=[ConstructionClaim(activity="Site clearance and earth leveling")],
    )
    with (
        patch("core.settings.settings.USE_FAKE_MODEL", False),
        patch.object(client, "_ainvoke_with_fallback", return_value=(model_output, "gemini-3.8-flash")),
    ):
        result = await client.extract_project_info(incomplete_ocr)

    assert isinstance(result, ExtractedProjectInfo)
    assert "site_coordinates" in result.missing_fields
    assert "project_dates" in result.missing_fields
    assert result.location.latitude is None
    assert result.location.longitude is None


@pytest.mark.asyncio
async def test_junk_strings_and_non_construction_routed_to_document_notes():
    """Test Item 6: Junk strings like dates and billing categories route to document_notes, NOT claims."""
    junk_ocr = """
    INVOICE & CLAIMS SUMMARY
    Project: Barasat Link Road Package 4
    Location: Latitude 22.7215, Longitude 88.4812
    Start Date: 2025-02-01
    Completion Date: 2026-09-30
    Claim 1: Period 01 January 2022 - 31 August 2024
    Claim 2: relate to the Transportation and logistics bills
    Claim 3: Bituminous road surfacing (4.5 km) - completed
    Claim 4: Reinforced concrete culvert construction - completed
    Billing Category: Overhead & Logistics Charges
    """

    client = GeminiVerificationClient(api_key="mock_key")
    result = await client.extract_project_info(junk_ocr)

    # Physical construction activities MUST be extracted as claims
    claim_activities = [c.activity.lower() for c in result.claims]
    assert any("road" in a or "surfacing" in a or "bituminous" in a for a in claim_activities)
    assert any("culvert" in a or "concrete" in a for a in claim_activities)

    # Non-construction items MUST NOT be claims
    assert not any("period 01" in a for a in claim_activities)
    assert not any("transportation" in a for a in claim_activities)
    assert not any("billing category" in a for a in claim_activities)

    # Non-construction items MUST be present in document_notes
    notes_lower = [n.lower() for n in result.document_notes]
    assert any("period 01" in n for n in notes_lower) or any("transportation" in n for n in notes_lower) or any("non-construction" in n for n in notes_lower)


@pytest.mark.asyncio
async def test_extraction_when_gemini_fails_does_not_pass_vacuously():
    """Test Item 10: State what is asserted when Gemini fails and ensure no trivial pass."""
    sample_text = """
    Project: NH-44 Bypass Works
    Location: Latitude 12.9716, Longitude 77.5946
    Start Date: 2024-01-10
    Completion Date: 2024-04-15
    Claim 1: Vegetation clearing and topsoil stripping - completed
    Claim 2: Period 01 January 2022 - 31 August 2024
    """

    client = GeminiVerificationClient(api_key="mock_key")
    # Simulate Gemini 429 quota exhaustion or 404: production must fail, not guess from the text.
    with (
        patch("core.settings.settings.USE_FAKE_MODEL", False),
        patch.object(client, "_ainvoke_with_fallback", side_effect=RuntimeError("429 RESOURCE_EXHAUSTED")),
    ):
        with pytest.raises(GeminiExtractionError, match="429"):
            await client.extract_project_info(sample_text)

    # The rule-based extractor remains available only in explicit test mode.
    with patch("core.settings.settings.USE_FAKE_MODEL", True):
        res = await client.extract_project_info(sample_text)

    assert isinstance(res, ExtractedProjectInfo)
    assert res.project_name == "NH-44 Bypass Works"
    assert res.location.latitude == 12.9716
    assert res.location.longitude == 77.5946
    assert res.start_date == "2024-01-10"
    assert res.completion_date == "2024-04-15"
    assert len(res.claims) == 1
    assert "clearing" in res.claims[0].activity.lower()
    # The junk claim must be filtered to document_notes
    assert not any("period 01" in c.activity.lower() for c in res.claims)
    assert any("period 01" in n.lower() or "non-construction" in n.lower() for n in res.document_notes)


def test_gemini_structured_response_validation():
    """Test 11: Gemini structured response schema validation with Pydantic."""
    raw_data = {
        "project_name": "Flyover Bridge Pier 12",
        "construction_type": "Bridge Infrastructure",
        "location": {
            "site_name": "Bridge Pier 12",
            "latitude": 19.0760,
            "longitude": 72.8777,
            "bbox": [72.8727, 19.0710, 72.8827, 19.0810],
        },
        "start_date": "2024-01-01",
        "completion_date": "2024-03-31",
        "claims": [
            {
                "activity": "Deep pile foundation drilling",
                "claimed_status": "completed",
                "quantity": "12 boreholes",
            }
        ],
        "missing_fields": [],
        "uncertain_fields": [],
    }

    validated = ExtractedProjectInfo.model_validate(raw_data)
    assert validated.project_name == "Flyover Bridge Pier 12"
    assert validated.claims[0].quantity == "12 boreholes"
    assert len(validated.location.bbox) == 4

