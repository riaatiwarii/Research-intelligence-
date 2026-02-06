# gap_detection/gap_analysis.py

import json
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest

CLEAN_DIR = Path("data/clean/papers")
COLLECTION = "papers"
TOP_K = 15
MODEL_NAME = "all-MiniLM-L6-v2"

qdrant = QdrantClient(host="localhost", port=6333)

def build_query_text(paper: dict) -> str:
    parts = []
    if paper.get("title"):
        parts.append(paper["title"])
    if paper.get("abstract"):
        parts.append(paper["abstract"])
    return "\n".join(parts).strip()

def query_neighbors(vector):
    res = qdrant.query_points(
        collection_name=COLLECTION,
        query=SearchRequest(
            vector=vector,
            limit=TOP_K,
            with_payload=True
        )
    )
    return res.points

def main():
    print("Running research gap analysis...")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    files = list(CLEAN_DIR.glob("*.json"))

    print("Papers loaded:", len(files))
    results = []

    for file in files:
        paper = json.loads(file.read_text(encoding="utf-8"))

        query_text = build_query_text(paper)
        if len(query_text) < 80:   # 🔥 relaxed
            continue

        vec = model.encode(query_text, normalize_embeddings=True).tolist()
        hits = query_neighbors(vec)

        scores = []
        for h in hits:
            pid = h.payload.get("paper_id")
            if pid and pid != paper.get("paper_id"):
                scores.append(h.score)

        if not scores:
            continue

        avg_sim = sum(scores) / len(scores)

        results.append({
            "paper_id": paper["paper_id"],
            "avg_similarity": round(avg_sim, 4)
        })

    results.sort(key=lambda x: x["avg_similarity"])

    print("\n=== TOP RESEARCH GAPS ===")
    for r in results[:10]:
        print(f"{r['paper_id']} → avg similarity {r['avg_similarity']}")

if __name__ == "__main__":
    main()
