import json
from pathlib import Path

import numpy as np
import hdbscan
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ---------------- CONFIG ----------------

CLEAN_DIR = Path("data/clean/papers")
COLLECTION = "papers"
MODEL_NAME = "all-MiniLM-L6-v2"

MIN_TEXT_LEN = 300
OUTLIER_THRESHOLD = 0.12   # <-- YOUR LINE (correct place)

# ---------------- INIT ----------------

qdrant = QdrantClient(host="localhost", port=6333)

# ---------------- HELPERS ----------------

def build_text(paper: dict) -> str:
    parts = []

    for key in ["title", "abstract", "text", "body_text", "full_text"]:
        if paper.get(key):
            parts.append(paper[key])

    if isinstance(paper.get("sections"), dict):
        parts.extend(
            v for v in paper["sections"].values()
            if isinstance(v, str)
        )

    return "\n".join(parts).strip()


# ---------------- MAIN ----------------

def main():
    print("Running HDBSCAN research gap detection...")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    vectors = []
    meta = []

    files = list(CLEAN_DIR.glob("*.json"))
    print("Papers loaded:", len(files))

    # ---------- EMBEDDING ----------

    for file in files:
        paper = json.loads(file.read_text(encoding="utf-8"))
        text = build_text(paper)

        if len(text) < MIN_TEXT_LEN:
            continue

        vec = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        vectors.append(vec)
        meta.append({
            "paper_id": paper.get("paper_id"),
            "source": paper.get("source")
        })

    if not vectors:
        print("No valid papers for clustering.")
        return

    X = np.vstack(vectors)

    # ---------- HDBSCAN ----------

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=4,
        metric="euclidean",
        prediction_data=True
    )

    labels = clusterer.fit_predict(X)
    outlier_scores = clusterer.outlier_scores_

    # ---------- GAP EXTRACTION ----------

    gaps = []

    for i, score in enumerate(outlier_scores):
        if labels[i] == -1:   # noise points only
            gaps.append({
                "paper_id": meta[i]["paper_id"],
                "source": meta[i]["source"],
                "outlier_score": float(score)
            })

    # ---------- FILTERING ----------
    SIGNIFICANT_GAPS = [
        g for g in gaps
        if g["outlier_score"] >= OUTLIER_THRESHOLD
    ]

    SIGNIFICANT_GAPS.sort(
        key=lambda x: x["outlier_score"],
        reverse=True
    )

    # ---------- OUTPUT ----------

    print("\n=== TOP RESEARCH GAPS (HDBSCAN) ===")
    for g in SIGNIFICANT_GAPS[:10]:
        print(
            f"Paper: {g['paper_id']} | "
            f"Source: {g['source']} | "
            f"Outlier score: {round(g['outlier_score'], 4)}"
        )

    print("\nTotal gap candidates:", len(gaps))
    print("Total SIGNIFICANT gaps:", len(SIGNIFICANT_GAPS))


if __name__ == "__main__":
    main()
