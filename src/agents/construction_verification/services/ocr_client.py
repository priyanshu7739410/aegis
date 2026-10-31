import asyncio
import io
import logging
from pathlib import Path
from typing import Any
import httpx
from PIL import Image
import pypdf
from pypdf.errors import PyPdfError

from core.settings import settings

logger = logging.getLogger(__name__)

OCR_SPACE_API_URL = "https://api.ocr.space/parse/image"
MAX_FILE_SIZE_BYTES = 1_500_000  # OCR.space free plan limit, enforced with HTTP 413
TARGET_IMAGE_BYTES = 1_400_000
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
# Scanned PDFs often embed a few stray characters; those still need real OCR.
MIN_EMBEDDED_TEXT_CHARS = 20
OCR_ATTEMPTS = 3
OCR_RETRY_DELAY_SECONDS = 2.0
TEMPORARY_OCR_FAILURES = ("timed out", "failed to communicate", "http 5", "non-json")


class OCRError(Exception):
    """Custom exception for OCR processing errors."""
    pass


class OCRClient:
    """Client for OCR.space, using embedded PDF text locally when it exists."""

    def __init__(self, api_key: str | None = None, timeout: float = 45.0):
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
        if suffix == ".pdf" and file_size > MAX_FILE_SIZE_BYTES:
            raise OCRError(
                f"'{path.name}' is {file_size / 1_000_000:.1f} MB and scanned, but the OCR.space "
                "free plan accepts at most 1.5 MB. Upload a smaller PDF or a photo of the page."
            )

        for attempt in range(1, OCR_ATTEMPTS + 1):
            try:
                return await self._call_ocr_space(path)
            except OCRError as e:
                temporary = any(k in str(e).lower() for k in TEMPORARY_OCR_FAILURES)
                if not temporary or attempt == OCR_ATTEMPTS:
                    raise
                logger.warning(f"OCR.space attempt {attempt}/{OCR_ATTEMPTS} failed: {e}")
                await asyncio.sleep(OCR_RETRY_DELAY_SECONDS * attempt)
        raise OCRError("OCR.space returned no result.")

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
            name, file_bytes = self._prepare_upload(path)
            files = {"file": (name, file_bytes)}
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

    @staticmethod
    def _prepare_upload(path: Path) -> tuple[str, bytes]:
        """Shrink oversized images to fit the OCR.space limit, keeping text legible."""
        content = path.read_bytes()
        if path.suffix.lower() == ".pdf" or len(content) <= MAX_FILE_SIZE_BYTES:
            return path.name, content

        with Image.open(io.BytesIO(content)) as source:
            image = source.convert("RGB")
        while True:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            if buffer.tell() <= TARGET_IMAGE_BYTES:
                return f"{path.stem}.jpg", buffer.getvalue()
            if min(image.size) < 500:
                raise OCRError(f"'{path.name}' could not be reduced below the OCR.space size limit.")
            image = image.resize((int(image.width * 0.8), int(image.height * 0.8)))

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
