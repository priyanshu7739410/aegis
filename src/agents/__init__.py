from typing import Any

__all__ = [
    "get_agent",
    "load_agent",
    "get_all_agent_info",
    "DEFAULT_AGENT",
    "AgentGraph",
    "AgentGraphLike",
]


def __getattr__(name: str) -> Any:
    # Lazy so importing agents.construction_verification does not load every registered agent.
    if name in __all__:
        from agents import agents as registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
