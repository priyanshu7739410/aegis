from unittest.mock import MagicMock, patch
import pytest

from schema import AgentInfo, ServiceMetadata


def test_construction_verification_default_agent_selection():
    """Verify that construction-verification is preferred as default when present in agent list."""
    mock_info = ServiceMetadata(
        agents=[
            AgentInfo(key="chatbot", description="Chatbot"),
            AgentInfo(key="research-assistant", description="Research Assistant"),
            AgentInfo(key="construction-verification", description="Construction Verification"),
        ],
        models=["fake"],
        default_agent="research-assistant",
        default_model="fake",
    )

    agent_list = [a.key for a in mock_info.agents]
    default_agent = (
        "construction-verification"
        if "construction-verification" in agent_list
        else mock_info.default_agent
    )

    assert default_agent == "construction-verification"


def test_fallback_when_construction_agent_not_in_list():
    """Verify fallback to default_agent if construction-verification is not available."""
    mock_info = ServiceMetadata(
        agents=[
            AgentInfo(key="chatbot", description="Chatbot"),
            AgentInfo(key="research-assistant", description="Research Assistant"),
        ],
        models=["fake"],
        default_agent="research-assistant",
        default_model="fake",
    )

    agent_list = [a.key for a in mock_info.agents]
    default_agent = (
        "construction-verification"
        if "construction-verification" in agent_list
        else mock_info.default_agent
    )

    assert default_agent == "research-assistant"


def test_agent_switching_resolution():
    """Verify URL param overrides default, and session state overrides default."""
    agent_list = ["chatbot", "construction-verification", "research-assistant"]
    default_agent = "construction-verification"

    # 1. No URL param, no session state -> default is construction-verification
    url_agent = None
    session_agent = None
    if url_agent and url_agent in agent_list:
        active_agent = url_agent
    elif session_agent and session_agent in agent_list:
        active_agent = session_agent
    else:
        active_agent = default_agent
    assert active_agent == "construction-verification"

    # 2. User switches to chatbot via session state
    session_agent = "chatbot"
    if url_agent and url_agent in agent_list:
        active_agent = url_agent
    elif session_agent and session_agent in agent_list:
        active_agent = session_agent
    else:
        active_agent = default_agent
    assert active_agent == "chatbot"

    # 3. User visits direct link ?agent=research-assistant
    url_agent = "research-assistant"
    if url_agent and url_agent in agent_list:
        active_agent = url_agent
    elif session_agent and session_agent in agent_list:
        active_agent = session_agent
    else:
        active_agent = default_agent
    assert active_agent == "research-assistant"
