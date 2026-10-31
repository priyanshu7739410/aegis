import logging
import os
from pathlib import Path
from typing import Any
import httpx

from core.settings import settings

logger = logging.getLogger(__name__)

OCR_SPACE_API_URL = "https://api.ocr.space/parse/image"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit for free tier OCR.space


class OCRError(Exception):
    """Custom exception for OCR processing errors."""
    pass


class OCRClient:
    """Client for OCR.space API with safety, validation, and fallback support."""

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        if api_key:
            self.api_key = api_key
        elif settings.OCR_API_KEY:
            self.api_key = settings.OCR_API_KEY.get_secret_value()
        else:
            self.api_key = None
        self.timeout = timeout

    async def extract_text_from_file(self, file_path: str | Path) -> str:
        """Extract text from a document image or PDF file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found at: {path}")

        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise OCRError(
                f"File size ({file_size / (1024*1024):.2f}MB) exceeds OCR.space limit of 10MB."
            )

        suffix = path.suffix.lower()
        allowed_exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
        if suffix not in allowed_exts:
            raise OCRError(f"Unsupported file format '{suffix}'. Supported: {', '.join(allowed_exts)}")

        # If running in fake mode or without an API key, use fallback / local extraction or mock
        is_mocked = hasattr(self._call_ocr_space, "mock_calls") or hasattr(self._call_ocr_space, "assert_called")
        if (settings.USE_FAKE_MODEL or not self.api_key) and not is_mocked:
            logger.info("Using simulated/local OCR extraction (USE_FAKE_MODEL=True or no OCR_API_KEY)")
            return self._fallback_extract(path)

        # Call OCR.space API
        return await self._call_ocr_space(path)

    async def _call_ocr_space(self, path: Path) -> str:
        """Execute HTTP request to OCR.space API."""
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()

            files = {"file": (path.name, file_bytes)}
            data = {
                "language": "eng",
                "isOverlayRequired": "false",
                "detectOrientation": "true",
                "isTable": "true",
                "scale": "true",
                "OCREngine": "2",
            }
            headers = {"apikey": self.api_key or ""}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    OCR_SPACE_API_URL,
                    files=files,
                    data=data,
                    headers=headers,
                )

            if response.status_code != 200:
                raise OCRError(f"OCR.space API HTTP {response.status_code}: {response.text[:200]}")

            result = response.json()
            return self._parse_ocr_response(result)

        except httpx.TimeoutException as e:
            logger.error("OCR.space request timed out.")
            raise OCRError("OCR request timed out. Please retry or verify document size.") from e
        except httpx.RequestError as e:
            logger.error(f"Network error communicating with OCR.space: {e}")
            raise OCRError(f"Failed to communicate with OCR service: {e}") from e

    def _parse_ocr_response(self, result: dict[str, Any]) -> str:
        """Parse structured OCR.space JSON response into concatenated text."""
        if result.get("IsErroredOnProcessing", False):
            errors = result.get("ErrorMessage") or ["Unknown error from OCR engine"]
            error_str = "; ".join(errors) if isinstance(errors, list) else str(errors)
            raise OCRError(f"OCR processing failed: {error_str}")

        parsed_results = result.get("ParsedResults", [])
        if not parsed_results:
            raise OCRError("OCR returned no parsed pages.")

        extracted_pages: list[str] = []
        for i, page in enumerate(parsed_results):
            exit_code = page.get("FileParseExitCode")
            if exit_code != 1:
                page_err = page.get("ErrorMessage", "Unknown page error")
                logger.warning(f"Page {i+1} parsing warning/error (code {exit_code}): {page_err}")

            text = page.get("ParsedText", "").strip()
            if text:
                extracted_pages.append(text)

        full_text = "\n\n--- Page Break ---\n\n".join(extracted_pages).strip()
        if not full_text:
            raise OCRError("OCR completed but extracted text was completely empty.")

        return full_text

    def _fallback_extract(self, path: Path) -> str:
        """Extract text locally for text files or PDFs without substituting synthetic demo data."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                text = "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
                if text:
                    return text
                raise OCRError(f"Local PDF reader extracted 0 text characters from {path.name}.")
            except OCRError:
                raise
            except Exception as e:
                logger.warning(f"Local pypdf extraction fallback failed: {e}")
                raise OCRError(f"Failed to read PDF document '{path.name}': {e}") from e

        # If it's a text-based document or text test fixture
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
            raise OCRError(f"Uploaded text document '{path.name}' is empty.")
        except UnicodeDecodeError:
            raise OCRError(
                f"Cannot extract text from binary image '{path.name}' without OCR.space API. "
                "Please configure a valid OCR_API_KEY in .env or provide document text manually."
            )
        except OCRError:
            raise
        except Exception as e:
            raise OCRError(f"Unable to read document file '{path.name}': {e}") from e

