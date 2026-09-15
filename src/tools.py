"""
tools.py
Defines every capability the agent is allowed to use.
Each tool is a plain Python function decorated with @tool so LangChain can
turn it into a schema the LLM understands (name, description, arguments).
"""

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from src import config 
from src.rag import document_search


search_tool = TavilySearchResults(max_results=3)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression, e.g. '23 * 4 + 1'.
    Only use this for math — never for anything else.
    """
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


TOOLS = [document_search, search_tool, calculator]