import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from indexing.vector_store import init_collection, upsert

CLEAN_DIR = Path("data/clean/papers")

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
OVERLAP = 100
MIN_LEN = 300

def chunk_text(text):
    words = text.split()
    step = CHUNK_SIZE - OVERLAP
    for i in range(0, len(words), step):
        yield " ".join(words[i:i + CHUNK_SIZE])

def build_text(paper: dict) -> str:
    parts = []

    # Title (important for retrieval)
    if paper.get("title"):
        parts.append(paper["title"])

    # Abstract
    if paper.get("abstract"):
        parts.append(paper["abstract"])

    # PMC-style sections
    if isinstance(paper.get("sections"), dict):
        for v in paper["sections"].values():
            if isinstance(v, str):
                parts.append(v)

    # arXiv-style body
    if paper.get("body_text"):
        parts.append(paper["body_text"])

    if paper.get("text"):
        parts.append(paper["text"])

    return "\n".join(parts).strip()


def main():
    print("Starting embedding job...")
    print("Clean dir:", CLEAN_DIR.resolve())

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"      # IMPORTANT
    )

    init_collection(dim=384)

    saved = 0
    skipped = 0
    total_chunks = 0

    files = list(CLEAN_DIR.glob("*.json"))
    print("Files found:", len(files))

    for idx, file in enumerate(files, 1):
        try:
            paper = json.loads(file.read_text(encoding="utf-8"))
            text = build_text(paper)

            if len(text) < MIN_LEN:
                skipped += 1
                continue

            for chunk in chunk_text(text):
                emb = model.encode(
                    chunk,
                    batch_size=1,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ).tolist()

                upsert(
                    emb,
                    {
                        "paper_id": paper.get("paper_id"),
                        "source": paper.get("source"),
                        "text": chunk
                    }
                )

                total_chunks += 1

            saved += 1

            if idx % 10 == 0:
                print(f"[{idx}/{len(files)}] papers embedded")

        except Exception as e:
            skipped += 1
            print(f"[ERROR] {file.name} | {e}")

    print("\n=== EMBEDDING SUMMARY ===")
    print("Saved papers :", saved)
    print("Skipped papers:", skipped)
    print("Total chunks :", total_chunks)

if __name__ == "__main__":
    main()
