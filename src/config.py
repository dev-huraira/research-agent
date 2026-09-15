

import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and injects its values into os.environ

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Guardrails / production limits
MAX_TOOL_CALLS = 6           # stop the agent after this many tool calls
MODEL_NAME = "gemini-3.5-flash-lite"
MODEL_TEMPERATURE = 0        # 0 = deterministic, best for agents following instructions precisely

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")