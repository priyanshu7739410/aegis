import logging
from pathlib import Path
from typing import Any
import httpx
import pypdf
from pypdf.errors import PyPdfError

from core.settings import settings

logger = logging.getLogger(__name__)

OCR_SPACE_API_URL = "https://api.ocr.space/parse/image"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit for free tier OCR.space
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
# Scanned PDFs often embed a few stray characters; those still need real OCR.
MIN_EMBEDDED_TEXT_CHARS = 20


class OCRError(Exception):
    """Custom exception for OCR processing errors."""
    pass


class OCRClient:
    """Client for OCR.space, using embedded PDF text locally when it exists."""

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

        suffix = path.suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise OCRError(
                f"Unsupported file format '{suffix}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        if suffix == ".pdf":
            local_text = self._extract_pdf_text(path)
            if local_text:
                return local_text

        if settings.USE_FAKE_MODEL:
            raise OCRError(
                f"'{path.name}' needs OCR.space, which is disabled while USE_FAKE_MODEL=true."
            )
        if not self.api_key:
            raise OCRError(f"'{path.name}' needs OCR.space but OCR_API_KEY is not configured.")

        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise OCRError(
                f"File size ({file_size / (1024*1024):.2f}MB) exceeds OCR.space limit of 10MB."
            )

        return await self._call_ocr_space(path)

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        """Return the PDF's embedded text, or an empty string when it needs real OCR."""
        try:
            reader = pypdf.PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except PyPdfError as e:
            logger.warning(f"Local PDF text extraction failed for {path.name}: {e}")
            return ""
        return text if len(text) >= MIN_EMBEDDED_TEXT_CHARS else ""

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

            try:
                result = response.json()
            except ValueError as e:
                raise OCRError(f"OCR.space returned a non-JSON response: {response.text[:200]}") from e
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
