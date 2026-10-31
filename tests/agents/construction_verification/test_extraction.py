import pytest
from unittest.mock import AsyncMock, patch

from agents.construction_verification.schemas import (
    ConstructionClaim,
    ExtractedProjectInfo,
    LocationInfo,
)
from agents.construction_verification.services.gemini_client import GeminiVerificationClient


@pytest.mark.asyncio
async def test_construction_claim_extraction():
    """Test 3: Construction claim extraction from document text into structured models."""
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

    client = GeminiVerificationClient()
    result = await client.extract_project_info(sample_ocr)

    assert isinstance(result, ExtractedProjectInfo)
    assert "Green Valley" in result.project_name or "Ring Road" in result.project_name
    assert result.location is not None
    assert pytest.approx(result.location.latitude, 0.01) == 13.0827
    assert pytest.approx(result.location.longitude, 0.01) == 80.2707
    assert result.start_date == "2024-02-01"
    assert result.completion_date == "2024-05-15"
    assert len(result.claims) >= 1
    assert any("clearing" in c.activity.lower() for c in result.claims)


@pytest.mark.asyncio
async def test_missing_location_and_dates_detection():
    """Test 4: Missing location coordinates and dates are accurately flagged."""
    incomplete_ocr = """
    GENERAL INVOICE
    The contractor claims that site clearance and earth leveling works have been finished.
    Payment requested.
    """

    client = GeminiVerificationClient()
    result = await client.extract_project_info(incomplete_ocr)

    assert isinstance(result, ExtractedProjectInfo)
    assert "site_coordinates" in result.missing_fields
    assert "project_dates" in result.missing_fields
    assert result.location.latitude is None
    assert result.location.longitude is None


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

