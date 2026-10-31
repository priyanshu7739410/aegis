"""Reproduce upload handling behavior with Gemini failing (404 and 429).

Demonstrates that the system returns a visible error or fallback result
within 60 seconds (zero hang).
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.construction_verification.services.gemini_client import GeminiVerificationClient


async def test_failure_mode(name: str, exception_to_raise: Exception):
    print(f"\n--- Testing Upload Extraction with Gemini {name} ---")
    client = GeminiVerificationClient(api_key="mock_key")
    
    sample_text = (
        "PROGRESS PAYMENT CLAIM\n"
        "Project: NH-44 Highway Expansion\n"
        "Location: Latitude 12.9716, Longitude 77.5946\n"
        "Start Date: 2024-01-10\n"
        "Claimed Completion Date: 2024-04-15\n"
        "Claim 1: Vegetation clearing and topsoil stripping (10,000 sq.m) - completed\n"
    )
    
    start_time = time.perf_counter()
    with patch.object(client, "_ainvoke_with_fallback", side_effect=exception_to_raise):
        res = await client.extract_project_info(sample_text)
    elapsed = time.perf_counter() - start_time
    
    print(f"Result type: {type(res).__name__}")
    print(f"Elapsed time: {elapsed:.3f} seconds (Must be < 60s)")
    print(f"Project name recovered via fallback: {res.project_name}")
    print(f"Claims recovered: {len(res.claims)}")
    print(f"Latitude recovered: {res.location.latitude}")
    assert elapsed < 60.0, f"Hang detected: took {elapsed}s >= 60s"
    assert len(res.claims) >= 1, "Fallback claims were not extracted"
    print(f"SUCCESS: System returned cleanly in {elapsed:.3f}s under Gemini {name}")


async def main():
    print("=== Reproducing Upload Behavior under Gemini 404 and 429 ===")
    
    # Mode 1: 404 Model Not Found
    err_404 = RuntimeError("404 NOT_FOUND. Model models/gemini-2.5-flash is no longer supported.")
    await test_failure_mode("404 NOT_FOUND", err_404)
    
    # Mode 2: 429 Quota Exhausted / Resource Exhausted
    err_429 = RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests.")
    await test_failure_mode("429 RESOURCE_EXHAUSTED", err_429)
    
    print("\n=== ALL UPLOAD FAILURE TESTS PASSED UNDER 60 SECONDS ===")


if __name__ == "__main__":
    asyncio.run(main())
