from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)
import uuid

COLLECTION_NAME = "paperlens_chunks"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dim

# Local file-based Qdrant — no cloud needed
client = QdrantClient(path="qdrant_db")


def init_collection():
    """Create collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"[VectorStore] Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"[VectorStore] Collection '{COLLECTION_NAME}' already exists.")


def store_chunks(chunks: list[str], vectors: list[list[float]], filename: str):
    """
    Store embedded chunks into Qdrant with filename as metadata.

    Each point stores:
    - vector: 384-dim embedding
    - payload: { text, filename }
    """
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, "filename": filename}
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[VectorStore] Stored {len(points)} chunks from '{filename}'.")


def search(query_vector: list[float], top_k: int = 20) -> list[dict]:
    """
    Search for top_k most similar chunks across ALL papers.

    Returns list of dicts with: text, filename, score
    """
    from qdrant_client.models import QueryRequest
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    return [
        {
            "text": r.payload["text"],
            "filename": r.payload["filename"],
            "score": round(r.score, 4)
        }
        for r in results.points
        if r.payload
    ]


def delete_by_filename(filename: str):
    """Delete all chunks belonging to a specific paper."""
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        )
    )
    print(f"[VectorStore] Deleted chunks for '{filename}'.")


def delete_all():
    """Wipe entire collection — used when user clears all papers."""
    client.delete_collection(COLLECTION_NAME)
    init_collection()
    print("[VectorStore] All chunks deleted, collection reset.")
