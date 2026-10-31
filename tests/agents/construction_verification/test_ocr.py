import io

import numpy as np
import pytest
from PIL import Image
from pathlib import Path
from unittest.mock import AsyncMock, patch
import httpx

from agents.construction_verification.services.ocr_client import (
    MAX_FILE_SIZE_BYTES,
    OCRClient,
    OCRError,
)


def test_ocr_response_parsing_multi_page():
    """Test 1: OCR response parsing from OCR.space multi-page JSON."""
    client = OCRClient(api_key="test-ocr-key")
    sample_response = {
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
        "ParsedResults": [
            {
                "FileParseExitCode": 1,
                "ParsedText": "Page 1: Contract Claim NH-44 Highway Expansion\nLatitude: 12.9716 Longitude: 77.5946",
                "ErrorMessage": "",
            },
            {
                "FileParseExitCode": 1,
                "ParsedText": "Page 2: Claimed Activities:\n1. Vegetation clearing completed\n2. Earthworks completed",
                "ErrorMessage": "",
            },
        ],
    }

    parsed_text = client._parse_ocr_response(sample_response)
    assert "Page 1: Contract Claim" in parsed_text
    assert "Page 2: Claimed Activities" in parsed_text
    assert "--- Page Break ---" in parsed_text


def test_ocr_api_failure_handling():
    """Test 2: OCR API failure detection when processing errored."""
    client = OCRClient(api_key="test-ocr-key")
    error_response = {
        "OCRExitCode": 3,
        "IsErroredOnProcessing": True,
        "ErrorMessage": ["File format unrecognized or corrupted."],
        "ParsedResults": [],
    }

    with pytest.raises(OCRError) as exc_info:
        client._parse_ocr_response(error_response)
    assert "OCR processing failed" in str(exc_info.value)


def test_ocr_empty_output_handling():
    """Test 2b: Handle empty OCR response."""
    client = OCRClient(api_key="test-ocr-key")
    empty_response = {
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
        "ParsedResults": [{"FileParseExitCode": 1, "ParsedText": "   "}],
    }

    with pytest.raises(OCRError) as exc_info:
        client._parse_ocr_response(empty_response)
    assert "completely empty" in str(exc_info.value)


def test_oversized_image_is_shrunk_to_fit_upload_limit(tmp_path):
    path = tmp_path / "phone_photo.png"
    noise = np.random.default_rng(1).integers(0, 255, (1500, 1500, 3), dtype=np.uint8)
    Image.fromarray(noise).save(path)
    assert path.stat().st_size > MAX_FILE_SIZE_BYTES

    name, content = OCRClient._prepare_upload(path)
    assert name == "phone_photo.jpg"
    assert len(content) <= MAX_FILE_SIZE_BYTES
    Image.open(io.BytesIO(content)).verify()


def test_small_file_is_uploaded_unchanged(tmp_path):
    path = tmp_path / "small.jpg"
    Image.new("RGB", (10, 10)).save(path)
    assert OCRClient._prepare_upload(path) == ("small.jpg", path.read_bytes())


@pytest.mark.asyncio
async def test_ocr_temporary_failure_is_retried(tmp_path):
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"png")
    client = OCRClient(api_key="test-key")
    call = AsyncMock(side_effect=[OCRError("OCR request timed out."), "recovered text"])
    with (
        patch.object(client, "_call_ocr_space", call),
        patch("core.settings.settings.USE_FAKE_MODEL", False),
        patch("asyncio.sleep", AsyncMock()),
    ):
        assert await client.extract_text_from_file(test_file) == "recovered text"
    assert call.await_count == 2


@pytest.mark.asyncio
async def test_ocr_permanent_failure_is_not_retried(tmp_path):
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"png")
    client = OCRClient(api_key="test-key")
    call = AsyncMock(side_effect=OCRError("OCR processing failed: invalid API key"))
    with (
        patch.object(client, "_call_ocr_space", call),
        patch("core.settings.settings.USE_FAKE_MODEL", False),
    ):
        with pytest.raises(OCRError, match="invalid API key"):
            await client.extract_text_from_file(test_file)
    assert call.await_count == 1


@pytest.mark.asyncio
async def test_ocr_network_timeout(tmp_path):
    """Test 2c: OCR network timeout."""
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"\x89PNG\r\n\x1a\nfakeimagebytes")

    client = OCRClient(api_key="test-key")
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Connection timed out")):
        with patch("core.settings.settings.USE_FAKE_MODEL", False):
            with pytest.raises(OCRError) as exc:
                await client.extract_text_from_file(test_file)
            assert "timed out" in str(exc.value)

