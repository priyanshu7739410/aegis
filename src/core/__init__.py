from typing import Any

from core.settings import settings

__all__ = ["settings", "get_model"]


def __getattr__(name: str) -> Any:
    # Lazy so the construction app can deploy without every LLM provider's SDK installed.
    if name == "get_model":
        from core.llm import get_model

        return get_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
