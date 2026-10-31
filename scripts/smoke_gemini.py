"""Smoke test for Gemini API multimodal and structured output capabilities.

Tests:
(a) One text call
(b) One image call using an existing rgb_after.png
(c) One with_structured_output call with a small Pydantic schema and an image

Usage:
    PYTHONPATH=src uv run python scripts/smoke_gemini.py
"""

import base64
import os
import sys
import time
from pathlib import Path
from pydantic import BaseModel, Field

# Ensure stdout flushes immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Ensure repo root and src are on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


class ImageSummary(BaseModel):
    is_satellite_or_aerial: bool = Field(description="True if image appears to be satellite or aerial imagery.")
    estimated_primary_land_cover: str = Field(description="Dominant visible land cover (e.g., bare ground, urban, agriculture, water).")
    brief_description: str = Field(description="1-2 sentence description of visible features.")


def find_sample_image() -> Path:
    candidates = list(REPO_ROOT.glob("outputs/**/rgb_after.png"))
    if not candidates:
        raise FileNotFoundError("Could not find any rgb_after.png in outputs/")
    # Prefer discovery run if present
    for p in candidates:
        if "discovery" in str(p):
            return p
    return candidates[0]


def invoke_with_retry(llm, message, max_retries=1, delay=1.0):
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(message)
        except Exception as e:
            msg = str(e).lower()
            if ("503" in msg or "unavailable" in msg) and attempt < max_retries:
                print(f"[Retry {attempt + 1}/{max_retries}] Transient 503 ({type(e).__name__}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def main():
    api_key_obj = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
    if not api_key_obj:
        print("ERROR: No GEMINI_API_KEY or GOOGLE_API_KEY configured in environment/settings.")
        sys.exit(1)

    api_key = api_key_obj.get_secret_value()
    # DO NOT PRINT OR LOG API KEY
    model_name = settings.GEMINI_MODEL
    fallback_model = settings.GEMINI_FALLBACK_MODEL

    print(f"=== Gemini Smoke Test ===")
    print(f"Configured GEMINI_MODEL: {model_name}")
    print(f"Configured GEMINI_FALLBACK_MODEL: {fallback_model}")

    # Helper to build LLM
    def get_chat_model(m_name):
        return ChatGoogleGenerativeAI(
            model=m_name,
            google_api_key=api_key,
            temperature=0.1,
            max_retries=0,
        )

    llm = get_chat_model(model_name)
    actual_model = model_name

    # --- (a) Text call ---
    print("\n--- (a) Text Call ---")
    text_prompt = "Reply in exactly five words confirming you are operational."
    try:
        res_a = invoke_with_retry(llm, text_prompt)
        content_a = getattr(res_a, "content", str(res_a))
        if isinstance(content_a, list) and content_a and isinstance(content_a[0], dict) and "text" in content_a[0]:
            content_a = content_a[0]["text"]
        print(f"Model used: {actual_model}")
        print(f"Response: {content_a}")
    except Exception as e:
        msg = str(e).lower()
        if ("404" in msg or "not found" in msg or "no longer supported" in msg) and fallback_model and fallback_model != model_name:
            print(f"Primary model {model_name} failed with 404/not-found: {e}")
            print(f"Falling back to {fallback_model}...")
            actual_model = fallback_model
            llm = get_chat_model(actual_model)
            res_a = invoke_with_retry(llm, text_prompt)
            content_a = getattr(res_a, "content", str(res_a))
            if isinstance(content_a, list) and content_a and isinstance(content_a[0], dict) and "text" in content_a[0]:
                content_a = content_a[0]["text"]
            print(f"Model used: {actual_model}")
            print(f"Response: {content_a}")
        else:
            print(f"Text call failed: {type(e).__name__}: {e}")
            sys.exit(1)

    # --- (b) Image call ---
    print("\n--- (b) Image Call ---")
    img_path = find_sample_image()
    print(f"Using image: {img_path.relative_to(REPO_ROOT)}")
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    img_msg = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the main ground features visible in this satellite patch in 1-2 sentences."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]
    )
    try:
        res_b = invoke_with_retry(llm, [img_msg])
        content_b = getattr(res_b, "content", str(res_b))
        if isinstance(content_b, list) and content_b and isinstance(content_b[0], dict) and "text" in content_b[0]:
            content_b = content_b[0]["text"]
        print(f"Model used: {actual_model}")
        print(f"Response: {content_b}")
    except Exception as e:
        print(f"Image call failed: {type(e).__name__}: {e}")
        sys.exit(1)

    # --- (c) with_structured_output call with image ---
    print("\n--- (c) Structured Output Call with Image ---")
    structured_llm = llm.with_structured_output(ImageSummary)
    struct_msg = HumanMessage(
        content=[
            {"type": "text", "text": "Extract structured details describing this satellite scene."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]
    )
    try:
        res_c = invoke_with_retry(structured_llm, [struct_msg])
        print(f"Model used: {actual_model}")
        print(f"Response: {res_c}")
    except Exception as e:
        print(f"Structured output call failed: {type(e).__name__}: {e}")
        sys.exit(1)

    print("\n=== Gemini Smoke Test COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()

