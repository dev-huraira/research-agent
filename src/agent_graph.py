"""
agent_graph.py
The ReAct loop, built as a graph.

Flow:
    START -> agent_node -> (decide) -> tools_node -> agent_node -> ... -> END
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg



from src.config import DATABASE_URL
from src.state import AgentState
from src.tools import TOOLS
from src.config import GOOGLE_API_KEY, MODEL_NAME, MODEL_TEMPERATURE, MAX_TOOL_CALLS

# 1. Create the LLM client.
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=MODEL_TEMPERATURE,
    google_api_key=GOOGLE_API_KEY,
)

# 2. Bind the tools to the LLM — tells the model these functions exist.
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "You are a research assistant. First check the document_search tool for relevant "
    "information in the local document collection. If it doesn't have what you need, "
    "use the search tool to find facts on the web. Use the calculator tool for any math. "
    "Do not guess numbers — look them up. Once you have enough information, give a "
    "direct final answer without calling more tools."
)


def agent_node(state: AgentState) -> dict:
    """The 'thinking' step."""
    messages = state["messages"]

    if not messages or messages[0].type != "system":
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if state["tool_call_count"] >= MAX_TOOL_CALLS:
        if getattr(last_message, "tool_calls", None):
            return "force_answer"   # wanted more tools but ran out of budget
        return "end"                # already had a real answer anyway

    if getattr(last_message, "tool_calls", None):
        return "continue"

    return "end"

def force_final_answer(state: AgentState) -> dict:
    """
    Runs only when we hit MAX_TOOL_CALLS. Asks the LLM to summarize
    with whatever information it already has, WITHOUT offering tools,
    so it's forced to produce real text instead of another tool request.
    """
    messages = state["messages"]
    response = llm.invoke(messages)  # note: llm, NOT llm_with_tools — no tools offered this time
    return {"messages": [response]}


def count_tool_calls(state: AgentState) -> dict:
    """Runs after tools execute, to update the safety counter."""
    last_ai_message = next(
        m for m in reversed(state["messages"]) if getattr(m, "tool_calls", None)
    )
    increment = len(last_ai_message.tool_calls)
    return {"tool_call_count": state["tool_call_count"] + increment}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("count", count_tool_calls)
    graph.add_node("force_answer", force_final_answer)   # new node

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END, "force_answer": "force_answer"},  # updated
    )

    graph.add_edge("tools", "count")
    graph.add_edge("count", "agent")
    graph.add_edge("force_answer", END)   # new edge

    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    memory = PostgresSaver(conn)
    memory.setup()
    return graph.compile(checkpointer=memory)


agent_app = build_graph()



