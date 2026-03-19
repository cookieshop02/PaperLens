from src.parser import extract_text_from_pdf, chunk_text
from src.embedder import embed_text
from src.vector_store import store_chunks, init_collection


def ingest_paper(file_bytes: bytes, filename: str) -> int:
    """
    Full ingestion pipeline for one PDF:
    1. Extract text
    2. Chunk it
    3. Embed all chunks
    4. Store in Qdrant

    Returns number of chunks stored.
    """
    print(f"[Ingest] Starting ingestion for '{filename}'...")

    # Step 1 — Extract
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        raise ValueError(f"Could not extract any text from '{filename}'. Is it a scanned PDF?")

    # Step 2 — Chunk
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"[Ingest] '{filename}' → {len(chunks)} chunks")

    # Step 3 — Embed all chunks in one batch (faster than one-by-one)
    vectors = embed_text(chunks).tolist()

    # Step 4 — Store
    init_collection()
    store_chunks(chunks, vectors, filename)

    print(f"[Ingest] Done — '{filename}' ingested with {len(chunks)} chunks.")
    return len(chunks)
