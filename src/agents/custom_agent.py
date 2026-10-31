"""Custom Agent Boilerplate.

This module provides a starting point for building a custom LangGraph agent.
Modify the system prompt, tools, state, and graph nodes to fit your use case.
"""

from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from core import get_model, settings


# 1. State Definition
class CustomAgentState(MessagesState, total=False):
    """Custom agent state extending MessagesState.

    Add any custom state fields here (e.g. user_id, context, flags).
    """

    custom_field: str


# 2. Tools Definition
@tool
def sample_tool(query: str) -> str:
    """A sample tool to demonstrate custom tool execution.

    Args:
        query: Input string or query for the tool.
    """
    return f"Sample tool output for '{query}' at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# List all tools available to this agent
tools = [sample_tool]
tool_node = ToolNode(tools)


# 3. System Instructions
SYSTEM_PROMPT = """You are a capable, intelligent AI assistant designed to solve user problems.
- Use the available tools when necessary.
- Provide structured, helpful, and concise answers.
- Cite sources or clearly explain calculations and logic when applicable.
"""


# 4. Agent Nodes
async def call_model(state: CustomAgentState, config: RunnableConfig) -> CustomAgentState:
    """Call the configured LLM with system prompt and tools."""
    # Retrieve configured model or fall back to default
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    model = get_model(model_name).bind_tools(tools)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = await model.ainvoke(messages, config)

    return {"messages": [response]}


def should_continue(state: CustomAgentState) -> Literal["tools", "done"]:
    """Determine whether to continue tool execution or finish."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "done"


# 5. Graph Assembly
builder = StateGraph(CustomAgentState)

builder.add_node("model", call_model)
builder.add_node("tools", tool_node)

builder.set_entry_point("model")
builder.add_conditional_edges(
    "model",
    should_continue,
    {
        "tools": "tools",
        "done": END,
    },
)
builder.add_edge("tools", "model")

# Compile the agent graph
custom_agent = builder.compile()

