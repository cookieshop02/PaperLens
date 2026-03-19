from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from src.ingest import ingest_paper
from src.rag_pipeline import run_pipeline
from src.vector_store import delete_all, init_collection
from src.database import (
    init_db, create_session, save_message,
    get_all_messages, get_last_n_messages, get_session
)

app = FastAPI(title="PaperLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Track uploaded papers in memory (per server run)
# Format: { filename: chunk_count }
uploaded_papers: dict[str, int] = {}

MAX_PAPERS = 4


# --- Startup ---

@app.on_event("startup")
def startup():
    init_db()
    init_collection()
    print("[PaperLens] Ready.")


# --- Health Check ---

@app.get("/")
def root():
    return {"message": "PaperLens is running 🔍"}


# --- Upload ---

@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    """Upload and ingest a research paper (PDF only, max 4 papers)."""

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Enforce max paper limit
    if len(uploaded_papers) >= MAX_PAPERS and file.filename not in uploaded_papers:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_PAPERS} papers allowed. Please clear papers before uploading new ones."
        )

    # Enforce file size (max 20MB)
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")

    try:
        chunk_count = ingest_paper(contents, file.filename)
        uploaded_papers[file.filename] = chunk_count
        return {
            "message": f"✅ '{file.filename}' ingested successfully.",
            "filename": file.filename,
            "chunks": chunk_count,
            "total_papers": len(uploaded_papers),
            "papers": list(uploaded_papers.keys())
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# --- List Papers ---

@app.get("/papers")
def list_papers():
    """List all currently uploaded papers."""
    return {
        "papers": list(uploaded_papers.keys()),
        "total": len(uploaded_papers),
        "slots_remaining": MAX_PAPERS - len(uploaded_papers)
    }


# --- Clear Papers ---

@app.delete("/papers")
def clear_papers():
    """Delete all uploaded papers and reset the vector store."""
    delete_all()
    uploaded_papers.clear()
    return {"message": "✅ All papers cleared. Vector store reset."}


# --- Chat ---

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Main Q&A endpoint.
    - Creates a new session if session_id not provided
    - Fetches last 5 messages for context
    - Runs RAG pipeline
    - Saves messages to PostgreSQL
    - Returns answer + full semantic breakdown
    """
    if not uploaded_papers:
        raise HTTPException(
            status_code=400,
            detail="No papers uploaded. Please upload research papers first."
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Create session if new
    session_id = request.session_id
    if not session_id:
        session_id = create_session(list(uploaded_papers.keys()))

    # Get last 5 messages for LLM context
    chat_history = get_last_n_messages(session_id, n=5)

    # Run RAG pipeline
    result = run_pipeline(request.query, chat_history)

    # Save to DB
    save_message(session_id, "user", request.query)
    save_message(session_id, "assistant", result["answer"])

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "top_chunks_overall": result["top_chunks_overall"],
        "top_chunks_per_paper": result["top_chunks_per_paper"],
        "per_paper_scores": result["per_paper_scores"]
    }


# --- Chat History ---

@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Fetch full chat history for a session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = get_all_messages(session_id)
    return {
        "session": session,
        "messages": messages,
        "total_messages": len(messages)
    }
