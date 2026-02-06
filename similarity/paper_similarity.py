import json
from pathlib import Path
import httpx
from sentence_transformers import SentenceTransformer

CLEAN_DIR = Path("data/clean/papers")
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "papers"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

def build_text(paper: dict) -> str:
    parts = []
    if paper.get("title"):
        parts.append(paper["title"])
    if paper.get("abstract"):
        parts.append(paper["abstract"])
    if isinstance(paper.get("sections"), dict):
        for v in paper["sections"].values():
            if isinstance(v, str):
                parts.append(v)
    if paper.get("full_text"):
        parts.append(paper["full_text"])
    return "\n".join(parts).strip()

def find_similar(query_text: str):
    vector = model.encode(
        query_text,
        normalize_embeddings=True
    ).tolist()

    payload = {
        "vector": vector,
        "limit": TOP_K,
        "with_payload": True
    }

    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search"

    with httpx.Client(timeout=60) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["result"]

def main():
    files = list(CLEAN_DIR.glob("*.json"))
    print("Files found:", len(files))

    target = files[0]
    print(f"\nFinding papers similar to: {target.name}\n")

    paper = json.loads(target.read_text(encoding="utf-8"))
    text = build_text(paper)

    results = find_similar(text)

    print("=== SIMILAR PAPERS ===")
    for i, r in enumerate(results, 1):
        print(
            f"{i}. Paper ID: {r['payload'].get('paper_id')} "
            f"| Score: {round(r['score'], 4)}"
        )

if __name__ == "__main__":
    main()
