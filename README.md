


# 🔍 PaperLens

> Upload research papers. Ask anything. See exactly where the answer came from.

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?logo=streamlit)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue?logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

🎉🎉🎉 PaperLens is live!
Here are URLs:

Frontend: https://huggingface.co/spaces/Cookie02Shop/paperlens
Backend: https://huggingface.co/spaces/Cookie02Shop/paperlens-api


## 📌 Table of Contents

- [About](#-about)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-reference)

---

## 📖 About

**PaperLens** is a RAG-based (Retrieval-Augmented Generation) research assistant that lets you upload up to 4 research papers and ask natural language questions across all of them simultaneously.

Unlike standard RAG chatbots that just return an answer, PaperLens shows the full semantic reasoning behind every response — which paper was most relevant, which exact chunks matched your query, and how confident the system is. This makes it a **transparent, trustworthy research assistant** rather than a black box.

**Real use cases:**
- Students asking questions across multiple research papers for a thesis
- Researchers comparing how different papers cover the same topic
- Anyone who wants to know *which paper* to cite, not just *what* the answer is

---

## 🧠 How It Works

```
  Your Question
       │
       ▼
┌──────────────────┐
│ all-MiniLM-L6-v2 │  ← embeds question into 384-dim vector
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     Qdrant       │  ← finds top-20 most similar chunks
│  (local file)    │     using cosine similarity
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Score Filter    │  ← drops chunks below 0.55 threshold
│  + Aggregation   │     aggregates per-paper scores
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Groq LLM      │  ← strict prompt: answer only from context
│ llama-3.1-8b     │     cites source papers in answer
└────────┬─────────┘
         │
         ▼
  Answer + Semantic Breakdown
  (per-paper scores, top chunks, confidence)
```

**Two phases:**

**Phase 1 — Ingestion** (on upload):
- Extract text from PDF
- Split into 500-char overlapping chunks
- Embed each chunk with `all-MiniLM-L6-v2`
- Store vectors + text in local Qdrant, tagged by filename

**Phase 2 — Q&A** (every query):
- Embed the question
- Search Qdrant for top-20 chunks across all papers
- Filter by confidence threshold
- Aggregate per-paper scores (max chunk score per paper)
- Send filtered chunks + chat history to Groq LLM
- Return answer + full semantic breakdown

---

## ✨ Features

- 📤 **Multi-PDF upload** — upload up to 4 research papers
- 💬 **Conversational Q&A** — multi-turn chat with memory (last 5 messages)
- 📊 **Per-paper similarity scores** — visual progress bars ranked by relevance
- 🏆 **Global top 3 chunks** — best matching passages across all papers
- 📄 **Per-paper chunk breakdown** — top 3 chunks per paper with expandable text
- ⚠️ **Low confidence warning** — alerts when query isn't covered in papers
- 🛡️ **Anti-hallucination system** — score thresholds + strict LLM prompting + source citation
- 🗄️ **Persistent chat history** — sessions and messages stored in PostgreSQL
- ⚡ **Fast inference** — Groq's `llama-3.1-8b-instant` for ultra-fast responses
- 🗃️ **Local vector store** — Qdrant runs fully locally, no cloud account needed

---

## 🛠️ Tech Stack

| Layer | Technology | Details |
|---|---|---|
| **Embeddings** | Sentence Transformers | `all-MiniLM-L6-v2` — 384-dim vectors |
| **Vector Database** | Qdrant | Local file-based storage |
| **LLM** | Groq | `llama-3.1-8b-instant` — ultra-fast inference |
| **Backend** | FastAPI | REST API, RAG pipeline orchestration |
| **Frontend** | Streamlit | Interactive chat + semantic breakdown UI |
| **Database** | PostgreSQL | Chat sessions and message history |
| **Language** | Python 3.10+ | Core language throughout |

---

## 📁 Project Structure

```
PaperLens/
│
├── src/
│   ├── main.py              # FastAPI app — all endpoints
│   ├── app.py               # Streamlit frontend
│   ├── rag_pipeline.py      # Orchestrates search → filter → aggregate → generate
│   ├── ingest.py            # PDF ingestion: parse → chunk → embed → store
│   ├── parser.py            # PDF text extraction + chunking
│   ├── embedder.py          # Sentence Transformers wrapper
│   ├── vector_store.py      # Qdrant client + search + delete
│   ├── llm.py               # Groq client + anti-hallucination prompt
│   └── database.py          # PostgreSQL — sessions + chat history
│
├── qdrant_db/               # Local Qdrant vector store (auto-generated)
│
├── .env                     # API keys — never commit this
├── .env.example             # Safe template
├── requirements.txt
├── docker-compose.yml       # PostgreSQL via Docker
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker Desktop
- A free [Groq API key](https://console.groq.com)

---

### 1. Clone the repository

```bash
git clone https://github.com/cookieshop02/PaperLens.git
cd PaperLens
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values (see [Environment Variables](#-environment-variables)).

### 5. Start PostgreSQL via Docker

```bash
docker-compose up -d
```

### 6. Start the FastAPI backend

```bash
uvicorn src.main:app --reload --port 8000
```

### 7. Start the Streamlit frontend

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` and start asking! 🎉

---

## 🔐 Environment Variables

Create a `.env` file in the root of the project:

```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://paperlens:paperlens123@localhost:5433/paperlens
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

---

## 📡 API Reference

Base URL: `http://localhost:8000`

---

### `GET /`
Health check.
```json
{ "message": "PaperLens is running 🔍" }
```

---

### `POST /upload`
Upload and ingest a research paper (PDF only, max 20MB).

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@paper.pdf"
```

```json
{
  "message": "✅ 'paper.pdf' ingested successfully.",
  "filename": "paper.pdf",
  "chunks": 142,
  "total_papers": 1,
  "papers": ["paper.pdf"]
}
```

---

### `POST /chat`
Ask a question against uploaded papers.

```json
{
  "query": "What is the transformer architecture?",
  "session_id": "optional-existing-session-id"
}
```

```json
{
  "session_id": "uuid",
  "answer": "The transformer architecture...[Source: paper.pdf]",
  "confidence": "high",
  "top_chunks_overall": [...],
  "top_chunks_per_paper": {...},
  "per_paper_scores": {
    "paper1.pdf": 0.91,
    "paper2.pdf": 0.61
  }
}
```

---

### `GET /papers`
List all currently uploaded papers.

---

### `DELETE /papers`
Clear all papers and reset the vector store.

---

### `GET /history/{session_id}`
Fetch full chat history for a session.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built with ❤️ by <a href="https://github.com/cookieshop02">cookieshop02</a></p>