"""Azure provider selection with a mocked client. No request is sent to Azure."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from pydantic import SecretStr

from agents.construction_verification.schemas import (
    ExtractedProjectInfo,
    VisionModelResponse,
)
from agents.construction_verification.services.gemini_client import (
    GeminiExtractionError,
    GeminiVerificationClient,
)

CLIENT = "agents.construction_verification.services.gemini_client"
CLAIMS = [{"activity": "Road surfacing", "claimed_status": "completed"}]


@pytest.fixture
def azure():
    with patch.multiple(
        "core.settings.settings",
        USE_FAKE_MODEL=False,
        CONSTRUCTION_LLM_PROVIDER="azure",
        CONSTRUCTION_AZURE_API_KEY=SecretStr("placeholder"),
        CONSTRUCTION_AZURE_ENDPOINT="https://fixture.invalid/",
        CONSTRUCTION_AZURE_DEPLOYMENT="text-deployment",
        CONSTRUCTION_AZURE_VISION_DEPLOYMENT="vision-deployment",
    ):
        yield


def _mocked_azure_class(result):
    azure_class = MagicMock()
    structured = azure_class.return_value.with_structured_output.return_value
    structured.ainvoke = AsyncMock(return_value=result)
    return azure_class, structured


def _images(tmp_path):
    paths = []
    for name in ("before.png", "after.png"):
        path = tmp_path / name
        Image.new("RGB", (8, 8)).save(path)
        paths.append(str(path))
    return paths


def test_extraction_uses_text_deployment_and_azure_settings(azure):
    output = ExtractedProjectInfo(project_name="Ring Road")
    azure_class, structured = _mocked_azure_class(output)
    with (
        patch(f"{CLIENT}.AzureChatOpenAI", azure_class),
        patch(f"{CLIENT}.ChatGoogleGenerativeAI") as gemini_class,
    ):
        result = asyncio.run(GeminiVerificationClient().extract_project_info("document"))

    assert result.project_name == "Ring Road"
    gemini_class.assert_not_called()
    kwargs = azure_class.call_args.kwargs
    assert kwargs["azure_deployment"] == "text-deployment"
    assert kwargs["azure_endpoint"] == "https://fixture.invalid/"
    assert kwargs["api_key"] == "placeholder"
    assert kwargs["max_retries"] == 0
    assert "temperature" not in kwargs
    azure_class.return_value.with_structured_output.assert_called_once_with(
        ExtractedProjectInfo, method="function_calling"
    )
    structured.ainvoke.assert_awaited_once()


def test_vision_uses_vision_deployment_and_records_it(azure, tmp_path):
    response = VisionModelResponse.model_validate(
        {
            "findings": [
                {"claim": "Road surfacing", "status": "SUPPORTED", "reasoning": "r",
                 "recommended_follow_up": "f"}
            ]
        }
    )
    azure_class, structured = _mocked_azure_class(response)
    before, after = _images(tmp_path)
    with patch(f"{CLIENT}.AzureChatOpenAI", azure_class):
        out = asyncio.run(
            GeminiVerificationClient().analyze_satellite_imagery(
                CLAIMS, before, after, {}, "2024-01-01", "2024-06-01"
            )
        )

    assert azure_class.call_args.kwargs["azure_deployment"] == "vision-deployment"
    content = structured.ainvoke.await_args.args[0][1].content
    assert len([part for part in content if part["type"] == "image_url"]) == 2
    assert out.findings[0].status == "SUPPORTED"
    assert out.vision_audit.vision_actually_executed is True
    assert out.vision_audit.model_used == "vision-deployment"


def test_vision_deployment_defaults_to_main_deployment(azure):
    with patch("core.settings.settings.CONSTRUCTION_AZURE_VISION_DEPLOYMENT", None):
        assert GeminiVerificationClient().vision_model == "text-deployment"


def test_azure_failure_produces_vision_not_executed_without_gemini_fallback(azure, tmp_path):
    azure_class, structured = _mocked_azure_class(None)
    structured.ainvoke = AsyncMock(side_effect=RuntimeError("503 unavailable"))
    before, after = _images(tmp_path)
    with (
        patch(f"{CLIENT}.AzureChatOpenAI", azure_class),
        patch(f"{CLIENT}.ChatGoogleGenerativeAI") as gemini_class,
        patch(f"{CLIENT}.asyncio.sleep", AsyncMock()),
    ):
        out = asyncio.run(
            GeminiVerificationClient().analyze_satellite_imagery(
                CLAIMS, before, after, {}, "2024-01-01", "2024-06-01"
            )
        )
    gemini_class.assert_not_called()
    assert structured.ainvoke.await_count == 4
    assert out.findings[0].status == "VISION_NOT_EXECUTED"
    assert out.vision_audit.vision_actually_executed is False


@pytest.mark.parametrize(
    "first_error",
    ["504 DEADLINE_EXCEEDED", "503 overloaded", "Request timed out.", "429 RESOURCE_EXHAUSTED"],
)
def test_temporary_failure_is_retried_and_recovers(azure, first_error):
    output = ExtractedProjectInfo(project_name="Ring Road")
    azure_class, structured = _mocked_azure_class(output)
    structured.ainvoke = AsyncMock(side_effect=[RuntimeError(first_error), output])
    with (
        patch(f"{CLIENT}.AzureChatOpenAI", azure_class),
        patch(f"{CLIENT}.asyncio.sleep", AsyncMock()) as sleep,
    ):
        result = asyncio.run(GeminiVerificationClient().extract_project_info("document"))
    assert result.project_name == "Ring Road"
    assert structured.ainvoke.await_count == 2
    sleep.assert_awaited_once()


def test_permanent_failure_is_not_retried(azure):
    azure_class, structured = _mocked_azure_class(None)
    structured.ainvoke = AsyncMock(side_effect=RuntimeError("401 invalid api key"))
    with patch(f"{CLIENT}.AzureChatOpenAI", azure_class):
        with pytest.raises(GeminiExtractionError, match="401"):
            asyncio.run(GeminiVerificationClient().extract_project_info("document"))
    assert structured.ainvoke.await_count == 1


@pytest.mark.parametrize(
    "setting,name",
    [
        ("CONSTRUCTION_AZURE_API_KEY", "CONSTRUCTION_AZURE_API_KEY"),
        ("CONSTRUCTION_AZURE_ENDPOINT", "CONSTRUCTION_AZURE_ENDPOINT"),
        ("CONSTRUCTION_AZURE_DEPLOYMENT", "CONSTRUCTION_AZURE_DEPLOYMENT"),
    ],
)
def test_incomplete_azure_config_fails_clearly(azure, setting, name):
    with patch(f"core.settings.settings.{setting}", None):
        with pytest.raises(GeminiExtractionError, match=name):
            asyncio.run(GeminiVerificationClient().extract_project_info("document"))


def test_gemini_remains_the_default_provider():
    from core.settings import settings

    assert settings.CONSTRUCTION_LLM_PROVIDER == "gemini"
    assert GeminiVerificationClient().provider == "gemini"
