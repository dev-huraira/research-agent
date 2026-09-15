from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

from src.config import GOOGLE_API_KEY

SAMPLE_DOCS = [
    Document(page_content="The research agent project uses LangGraph to build a ReAct-style loop with search and calculator tools."),
    Document(page_content="Huraira is a final-year BS Information Technology student specializing in NLP and LLM engineering."),
    Document(page_content="The project uses Google's Gemini model instead of Anthropic's Claude, accessed via the langchain-google-genai package."),
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

vector_store = Chroma.from_documents(
    documents=SAMPLE_DOCS,
    embedding=embeddings,
    collection_name="research_agent_docs",
)


retriever = vector_store.as_retriever(search_kwargs={"k": 2})

@tool
def document_search(query: str) -> str:
    """
    Search your own document collection for relevant information.
    Always try this BEFORE using the web search tool — it's faster
    and more trustworthy for anything already in your documents.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant documents found in the local collection."
    return "\n\n".join(doc.page_content for doc in results)


