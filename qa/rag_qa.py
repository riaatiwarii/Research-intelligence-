import json
from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests

# =========================
# CONFIG
# =========================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "papers"

EMBED_MODEL = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"   # or: mistral, phi3, gemma

TOP_K = 5
MAX_CONTEXT_CHARS = 6000


# =========================
# INIT
# =========================
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

qdrant = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)


# =========================
# VECTOR SEARCH
# =========================
def retrieve_context(query: str, k: int = TOP_K) -> List[str]:
    query_emb = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_emb,
        limit=k
    )

    chunks = []
    for r in results:
        payload = r.payload or {}
        text = payload.get("text", "")
        if text:
            chunks.append(text)

    return chunks


# =========================
# PROMPT BUILDING
# =========================
def build_prompt(question: str, contexts: List[str]) -> str:
    context_text = "\n\n---\n\n".join(contexts)
    context_text = context_text[:MAX_CONTEXT_CHARS]

    return f"""
You are a research assistant.

Answer the question strictly using the provided research context.
If the answer is not present, say "Not enough evidence in the provided papers."

Context:
{context_text}

Question:
{question}

Answer:
""".strip()


# =========================
# OLLAMA CALL
# =========================
def generate_answer(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()
    return response.json()["response"].strip()


# =========================
# MAIN LOOP
# =========================
def main():
    print("\nRAG QA READY")
    print("Type 'exit' to quit\n")

    while True:
        question = input(">> Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        print("Retrieving relevant papers...")
        contexts = retrieve_context(question)

        if not contexts:
            print("\nNo relevant context found.\n")
            continue

        prompt = build_prompt(question, contexts)

        print("Generating answer...\n")
        answer = generate_answer(prompt)

        print("===== ANSWER =====")
        print(answer)
        print("==================\n")


if __name__ == "__main__":
    main()
