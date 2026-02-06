import json
from pathlib import Path

RAW_DIR = Path("data/processed/papers_json")
OUT_DIR = Path("data/clean/papers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS = 500


def extract_text(paper: dict) -> str | None:
    """
    Normalize all sources into a single text field.
    """
    if "full_text" in paper and paper["full_text"]:
        return paper["full_text"]

    if "sections" in paper and isinstance(paper["sections"], dict):
        sections = []
        for _, sec_text in paper["sections"].items():
            if isinstance(sec_text, str):
                sections.append(sec_text)
        if sections:
            return "\n\n".join(sections)

    return None


def run():
    saved = 0
    skipped = 0

    for json_file in RAW_DIR.glob("*.json"):
        try:
            paper = json.loads(json_file.read_text(encoding="utf-8"))

            paper_id = paper.get("paper_id")
            source = paper.get("source")

            if not paper_id or not source:
                skipped += 1
                continue

            text = extract_text(paper)

            if not text or len(text) < MIN_CHARS:
                skipped += 1
                continue

            clean_paper = {
                "paper_id": paper_id,
                "source": source,
                "text": text,
                "char_count": len(text)
            }

            out_file = OUT_DIR / f"{paper_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(clean_paper, f, indent=2, ensure_ascii=False)

            saved += 1

        except Exception:
            skipped += 1

    print("CLEAN DATASET SUMMARY")
    print("Saved:", saved)
    print("Skipped:", skipped)


if __name__ == "__main__":
    run()
