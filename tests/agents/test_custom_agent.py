import pytest
from langchain_core.messages import HumanMessage

from agents.agents import get_agent
from agents.custom_agent import custom_agent, sample_tool


def test_custom_agent_registration():
    """Verify custom agent is registered and retrievable."""
    agent = get_agent("custom-agent")
    assert agent is not None
    assert agent == custom_agent


def test_sample_tool():
    """Verify the sample tool executes properly."""
    result = sample_tool.invoke({"query": "Antigravity"})
    assert "Sample tool output for 'Antigravity'" in result


@pytest.mark.xfail(reason="Upstream toolkit test requires live OPENAI_API_KEY")
@pytest.mark.asyncio
async def test_custom_agent_invoke():
    """Test invoking the custom agent graph."""
    inputs = {"messages": [HumanMessage(content="Hello, test assistant!")]}
    result = await custom_agent.ainvoke(inputs)
    assert "messages" in result
    assert len(result["messages"]) > 0

