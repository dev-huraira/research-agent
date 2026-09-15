from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

from src.agent_graph import agent_app
from src.state import AgentState


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research_agent")

app = FastAPI(title="Research Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",           # local Next.js dev server
        "https://your-frontend.vercel.app", # replace with your real Vercel URL once you have it
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = "default"


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty or whitespace-only.")

    start_time = time.time()
    config = {"configurable": {"thread_id": request.thread_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.message)],
        "tool_call_count": 0,
    }

    result = agent_app.invoke(initial_state, config=config)
    final_message = result["messages"][-1]

    content = final_message.content
    if isinstance(content, list):
        answer_text = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    else:
        answer_text = content

    elapsed = time.time() - start_time
    logger.info(
        "thread_id=%s tool_calls=%d elapsed=%.2fs",
        request.thread_id,
        result["tool_call_count"],
        elapsed,
    )

    return {"answer": answer_text, "tool_calls_used": result["tool_call_count"]}


@app.get("/health")
def health():
    return {"status": "ok"}