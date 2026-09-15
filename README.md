# Research Agent (LangGraph, production-scaffold)

## What this is
A ReAct-style agent (search + calculator tools) built as a LangGraph
state machine, with checkpointed memory and a FastAPI serving layer.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your real API keys
```

Get keys:
- Anthropic: https://console.anthropic.com
- Tavily (free tier): https://tavily.com
- LangSmith (optional, for tracing): https://smith.langchain.com

## Run the API
```bash
uvicorn src.main:app --reload
```

Test it:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the population of Pakistan divided by 1000?", "thread_id": "test1"}'
```

Send another message with the SAME `thread_id` and the agent will
remember the earlier turn (thanks to the SQLite checkpointer).

## Project structure
```
research_agent/
├── .env.example
├── requirements.txt
├── src/
│   ├── config.py       # env vars + guardrail constants
│   ├── state.py         # shape of data flowing through the graph
│   ├── tools.py         # search + calculator tool definitions
│   ├── agent_graph.py   # the ReAct loop as a LangGraph StateGraph
│   └── main.py          # FastAPI wrapper
└── tests/
```

## Next steps (not yet built)
- Add a RAG retriever tool over your own documents
- Swap SqliteSaver -> PostgresSaver for real deployment
- Add LangSmith tracing (set LANGCHAIN_TRACING_V2=true in .env)
- Write eval test cases in tests/
