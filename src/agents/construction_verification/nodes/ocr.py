import logging
from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.construction_verification.services.ocr_client import OCRClient, OCRError
from agents.construction_verification.state import ConstructionVerificationState

logger = logging.getLogger(__name__)


async def ocr_node(
    state: ConstructionVerificationState, config: RunnableConfig
) -> dict[str, Any]:
    """Extract text from uploaded construction documents using OCR.space."""
    extracted_text = state.get("extracted_text", "").strip()
    document_paths = state.get("document_paths", [])
    errors = list(state.get("errors", []))

    if extracted_text:
        logger.info("OCR text already provided in state. Skipping OCR step.")
        return {"extracted_text": extracted_text}

    if not document_paths:
        if state.get("construction_claims") or state.get("project_name"):
            logger.info("Project claims/metadata already present in state without documents.")
            return {}
        msg = "No construction documents provided for OCR processing."
        logger.warning(msg)
        return {
            "errors": errors + [msg],
            "messages": [AIMessage(content="Please provide a construction document (PDF or image) to verify claims.")],
        }

    ocr_client = OCRClient()
    accumulated_texts: list[str] = []

    for doc_path in document_paths:
        try:
            logger.info(f"Running OCR on document: {doc_path}")
            text = await ocr_client.extract_text_from_file(doc_path)
            accumulated_texts.append(text)
        except OCRError as e:
            err_msg = f"OCR failed for {doc_path}: {e}"
            logger.error(err_msg)
            errors.append(err_msg)
        except Exception as e:
            err_msg = f"Unexpected error during OCR on {doc_path}: {e}"
            logger.error(err_msg)
            errors.append(err_msg)

    final_text = "\n\n=== NEXT DOCUMENT ===\n\n".join(accumulated_texts).strip()

    if not final_text:
        err = "Could not extract any text from the provided documents."
        return {
            "errors": errors + [err],
            "ocr_error": "; ".join(errors + [err]),
            "messages": [AIMessage(content=f"Error extracting document text: {'; '.join(errors)}")],
        }

    return {
        "extracted_text": final_text,
        "ocr_error": None,
        "errors": errors,
        "messages": [AIMessage(content=f"Successfully extracted {len(final_text)} characters from {len(document_paths)} document(s).")],
    }

