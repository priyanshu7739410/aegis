"""Explicit test fixtures standing in for uploads, STAC and Gemini responses."""

import io

from PIL import Image


def png_bytes(shade: int = 128) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, shade, shade)).save(buffer, format="PNG")
    return buffer.getvalue()


def scene_fixture(scene_id: str, acquisition_date: str) -> dict:
    bands = ("red", "green", "blue", "nir", "scl", "swir16")
    return {
        "scene_id": scene_id,
        "acquisition_date": acquisition_date,
        "cloud_cover_percent": 3.0,
        "asset_urls": {
            band: f"https://stac-fixture.invalid/{scene_id}/{band}.tif" for band in bands
        },
        "bbox": [80.0, 13.0, 81.0, 14.0],
        "properties": {"eo:cloud_cover": 3.0},
    }


def vision_response(claim: str, status: str) -> dict:
    return {
        "visible_progress": "VISIBLE_PROGRESS",
        "progress_statement": "fixture progress statement",
        "findings": [
            {
                "claim": claim,
                "status": status,
                "visual_observations": ["fixture observation"],
                "reasoning": "fixture reasoning",
                "limitations": ["fixture limitation"],
                "recommended_follow_up": "fixture follow-up",
            }
        ],
        "general_observations": "fixture general observation",
    }
