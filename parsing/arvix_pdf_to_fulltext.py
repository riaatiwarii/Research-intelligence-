import json
from pathlib import Path
from pdfminer.high_level import extract_text
import unicodedata

RAW_ARXIV = Path("data/raw/arxiv")
OUT_DIR = Path("data/processed/papers_json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS = 500
LIMIT = 300


def clean_unicode(text: str) -> str:
    # Normalize unicode (α → alpha-like safe form where possible)
    text = unicodedata.normalize("NFKC", text)
    # Remove null bytes
    return text.replace("\x00", "")


def run():
    saved, skipped = 0, 0

    for i, pdf in enumerate(RAW_ARXIV.glob("*.pdf")):
        if i >= LIMIT:
            break

        out_file = OUT_DIR / f"{pdf.stem}.json"
        if out_file.exists():
            continue

        print(f"Processing (pdfminer): {pdf.name}")

        try:
            text = extract_text(pdf)
            if not text:
                print("[SKIP] No text extracted")
                skipped += 1
                continue

            text = clean_unicode(text)

            if len(text) < MIN_CHARS:
                print(f"[SKIP] Too short ({len(text)} chars)")
                skipped += 1
                continue

            paper = {
                "paper_id": pdf.stem,
                "source": "arxiv",
                "full_text": text,
                "char_count": len(text)
            }

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(paper, f, indent=2, ensure_ascii=False)

            print(f"[OK] Saved ({len(text)} chars)")
            saved += 1

        except Exception as e:
            print(f"[ERROR] {pdf.name} | {e}")
            skipped += 1

    print("\nSUMMARY")
    print("Saved:", saved)
    print("Skipped:", skipped)


if __name__ == "__main__":
    run()
