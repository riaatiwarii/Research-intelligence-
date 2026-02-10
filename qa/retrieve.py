# qa/retrieve.py

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import List, Dict
import os

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "papers"

model = SentenceTransformer(MODEL_NAME)

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333")
)


def retrieve_chunks(query: str, top_k: int = 8) -> List[Dict]:
    query_embedding = model.encode(query).tolist()

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )

    chunks = []
    for r in response.points:   # ✅ THIS IS THE KEY FIX
        payload = r.payload
        chunks.append({
            "text": payload["text"],
            "paper_id": payload["paper_id"],
            "section": payload.get("section_type", "unknown"),
            "score": r.score
        })

    return chunks
