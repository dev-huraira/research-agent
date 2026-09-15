"""
test_agent.py
A small evaluation suite: checks that the agent picks the right tool
for different kinds of questions, and that memory persists correctly.

Note: these are REAL integration tests — they call the actual Gemini
and Tavily APIs, so they cost real (tiny) amounts of quota and take
a few seconds each. Run them before shipping any change to the agent's
prompt, tools, or graph structure.
"""

import pytest
from langchain_core.messages import HumanMessage

from src.agent_graph import agent_app
from src.state import AgentState


def run_agent(message: str, thread_id: str) -> dict:
    """Helper: runs one turn and returns the full result state."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AgentState = {
        "messages": [HumanMessage(content=message)],
        "tool_call_count": 0,
    }
    return agent_app.invoke(initial_state, config=config)


def get_tool_names_used(result: dict) -> list[str]:
    """Helper: pulls out which tool names were actually called during this run."""
    return [
        call["name"]
        for msg in result["messages"]
        if getattr(msg, "tool_calls", None)
        for call in msg.tool_calls
    ]


def test_math_question_uses_calculator():
    result = run_agent("What is 4582 multiplied by 917?", thread_id="eval_math")
    tools_used = get_tool_names_used(result)
    assert "calculator" in tools_used
    final_answer = result["messages"][-1].content
    assert "4201694" in str(final_answer)  # the actual correct product


def test_current_events_question_uses_search():
    result = run_agent("What is the current population of Pakistan?", thread_id="eval_search")
    tools_used = get_tool_names_used(result)
    assert "tavily_search_results_json" in tools_used


def test_local_docs_question_uses_rag_tool():
    result = run_agent("What model does this research agent project use?", thread_id="eval_rag")
    tools_used = get_tool_names_used(result)
    assert "document_search" in tools_used


def test_memory_persists_across_turns():
    thread_id = "eval_memory"
    run_agent("My favorite country to study is Pakistan.", thread_id=thread_id)
    result = run_agent("What country did I just mention?", thread_id=thread_id)
    final_answer = str(result["messages"][-1].content).lower()
    assert "pakistan" in final_answer


def test_empty_conversation_does_not_crash():
    # Sanity check: a trivial question that needs NO tools should still work
    result = run_agent("Say hello in one word.", thread_id="eval_trivial")
    final_answer = result["messages"][-1].content
    assert final_answer  # not empty


def extract_answer_text(result: dict) -> str:
    """Same content-extraction logic as main.py — content can be a plain
    string or a list of content blocks depending on the model's response."""
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


def test_math_question_uses_calculator():
    result = run_agent("What is 4582 multiplied by 917?", thread_id="eval_math")
    tools_used = get_tool_names_used(result)
    assert "calculator" in tools_used

    final_answer = extract_answer_text(result)
    final_answer_no_commas = final_answer.replace(",", "")
    assert "4201694" in final_answer_no_commas