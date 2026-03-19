from src.embedder import embed_query
from src.vector_store import search
from src.llm import generate_answer

# Thresholds
MIN_SCORE_TO_ANSWER = 0.45   # below this → warn user, don't call LLM
MIN_SCORE_FOR_CONTEXT = 0.55  # only pass chunks above this to LLM


def aggregate_per_paper(chunks: list[dict]) -> dict[str, float]:
    """
    For each paper, take the MAX score across all its chunks.
    Max score is more meaningful than average for relevance ranking.
    """
    paper_scores = {}
    for chunk in chunks:
        fname = chunk["filename"]
        score = chunk["score"]
        if fname not in paper_scores or score > paper_scores[fname]:
            paper_scores[fname] = score

    # Sort by score descending
    return dict(sorted(paper_scores.items(), key=lambda x: x[1], reverse=True))


def get_top_chunks_overall(chunks: list[dict], n: int = 3) -> list[dict]:
    """Return top N chunks by score across all papers."""
    sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
    return [
        {"rank": i + 1, "filename": c["filename"], "score": c["score"], "text": c["text"]}
        for i, c in enumerate(sorted_chunks[:n])
    ]


def get_top_chunks_per_paper(chunks: list[dict], n: int = 3) -> dict[str, list[dict]]:
    """For each paper, return its top N chunks by score."""
    paper_chunks: dict[str, list] = {}

    for chunk in chunks:
        fname = chunk["filename"]
        if fname not in paper_chunks:
            paper_chunks[fname] = []
        paper_chunks[fname].append(chunk)

    result = {}
    for fname, pchunks in paper_chunks.items():
        sorted_pchunks = sorted(pchunks, key=lambda x: x["score"], reverse=True)
        result[fname] = [
            {"score": c["score"], "text": c["text"]}
            for c in sorted_pchunks[:n]
        ]

    return result


def run_pipeline(query: str, chat_history: list[dict]) -> dict:
    """
    Full RAG pipeline:
    1. Embed query
    2. Search Qdrant (top 20 chunks)
    3. Check confidence
    4. Filter chunks for LLM context
    5. Aggregate per-paper scores
    6. Generate answer
    7. Return structured response

    Returns:
        {
            answer, confidence, top_chunks_overall,
            top_chunks_per_paper, per_paper_scores
        }
    """

    # Step 1 — Embed query
    query_vector = embed_query(query)

    # Step 2 — Search
    all_chunks = search(query_vector, top_k=20)

    if not all_chunks:
        return {
            "answer": "No papers have been uploaded yet. Please upload research papers first.",
            "confidence": "none",
            "top_chunks_overall": [],
            "top_chunks_per_paper": {},
            "per_paper_scores": {}
        }

    # Step 3 — Confidence check
    best_score = all_chunks[0]["score"]
    if best_score < MIN_SCORE_TO_ANSWER:
        return {
            "answer": f"⚠️ I couldn't find relevant content in the uploaded papers for this query. Best match score was only {best_score:.2f}. Try rephrasing or check if this topic is covered.",
            "confidence": "low",
            "top_chunks_overall": get_top_chunks_overall(all_chunks),
            "top_chunks_per_paper": get_top_chunks_per_paper(all_chunks),
            "per_paper_scores": aggregate_per_paper(all_chunks)
        }

    # Step 4 — Filter chunks for LLM (only strong matches)
    context_chunks = [c for c in all_chunks if c["score"] >= MIN_SCORE_FOR_CONTEXT]

    # Fallback — if nothing passes threshold but best_score was okay, take top 5
    if not context_chunks:
        context_chunks = all_chunks[:5]

    # Step 5 — Aggregate scores
    per_paper_scores = aggregate_per_paper(all_chunks)

    # Step 6 — Generate answer
    answer = generate_answer(query, context_chunks, chat_history)

    # Step 7 — Build response
    confidence = "high" if best_score >= 0.7 else "medium"

    return {
        "answer": answer,
        "confidence": confidence,
        "top_chunks_overall": get_top_chunks_overall(all_chunks),
        "top_chunks_per_paper": get_top_chunks_per_paper(all_chunks),
        "per_paper_scores": per_paper_scores
    }
