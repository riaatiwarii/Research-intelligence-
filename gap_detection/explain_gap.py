import json
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ---------------- CONFIG ----------------

CLEAN_DIR = Path("data/clean/papers")
COLLECTION = "papers"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 6

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

qdrant = QdrantClient(host="localhost", port=6333)

# ---------------- HELPERS ----------------

def build_focus_text(paper):
    parts = []

    if paper.get("title"):
        parts.append(paper["title"])
    if paper.get("abstract"):
        parts.append(paper["abstract"])
    if paper.get("body_text"):
        parts.append(paper["body_text"][:1500])
    if paper.get("text"):
        parts.append(paper["text"][:1500])

    return "\n".join(parts).strip()


def query_chunks(vector):
    res = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=TOP_K,
        with_payload=True
    )
    return res.points


def explain_gap_llm(focus, neighbors):
    prompt = f"""
You are a research analyst.

Paper focus:
{focus[:600]}

Related literature:
{neighbors[:800]}

Task:
Explain the research gap in 3–4 sentences.
"""

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 200,
                    "temperature": 0.4
                }
            },
            timeout=300   # IMPORTANT
        )

        r.raise_for_status()
        return r.json()["response"].strip()

    except requests.exceptions.ReadTimeout:
        return "[TIMEOUT] Ollama did not respond in time."

# ---------------- MAIN ----------------

def main():
    print("Explaining research gaps...\n")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    files = list(CLEAN_DIR.glob("*.json"))
    print("Usable papers:", len(files))

    explained = 0

    for file in files[:10]:  # LIMIT intentionally
        paper = json.loads(file.read_text(encoding="utf-8"))

        focus_text = build_focus_text(paper)
        if len(focus_text) < 200:
            continue

        vec = model.encode(
            focus_text,
            normalize_embeddings=True
        ).tolist()

        hits = query_chunks(vec)

        neighbor_chunks = [
            h.payload["text"]
            for h in hits
            if h.payload and "text" in h.payload
        ]

        if not neighbor_chunks:
            continue

        explanation = explain_gap_llm(
            focus_text[:1000],
            "\n\n".join(neighbor_chunks[:3])
        )

        print("=" * 70)
        print(f"Paper: {paper.get('paper_id')} | Source: {paper.get('source')}\n")
        print(explanation)

        explained += 1

    print(f"\nExplained gaps for {explained} papers.")

if __name__ == "__main__":
    main()
