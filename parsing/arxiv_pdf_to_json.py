import requests
import json
import time
from pathlib import Path
from lxml import etree

# ----------------------------------
# PATHS
# ----------------------------------
RAW_ARXIV = Path("data/raw/arxiv")
OUT_DIR = Path("data/processed/papers_json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# GROBID CONFIG (AGGRESSIVE)
# ----------------------------------
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

REQUEST_TIMEOUT = 120   # 2 minutes max per PDF
RETRIES = 0             # DO NOT retry (avoid stalls)
BATCH_LIMIT = 200       # HARD STOP after 200 PDFs


# ----------------------------------
# GROBID CALL
# ----------------------------------
def parse_with_grobid(pdf_path):
    with open(pdf_path, "rb") as f:
        response = requests.post(
            GROBID_URL,
            files={"input": f},
            timeout=REQUEST_TIMEOUT
        )
    response.raise_for_status()
    return response.text


# ----------------------------------
# TEI XML → JSON
# ----------------------------------
def tei_to_json(tei_xml, paper_id):
    root = etree.fromstring(tei_xml.encode())

    def extract(xpath):
        return " ".join(root.xpath(xpath)).strip()

    paper = {
        "paper_id": paper_id,
        "source": "arxiv",
        "title": extract("//titleStmt/title//text()"),
        "abstract": extract("//abstract//text()"),
        "sections": {}
    }

    for div in root.xpath("//text/body/div"):
        head = div.findtext("head")
        if not head:
            continue

        content = " ".join(div.xpath(".//p//text()")).strip()
        if len(content) < 200:
            continue

        paper["sections"][head.lower()] = content

    return paper


# ----------------------------------
# RUNNER
# ----------------------------------
def run():
    parsed = 0
    skipped = 0

    for idx, pdf in enumerate(RAW_ARXIV.glob("*.pdf")):
        if idx >= BATCH_LIMIT:
            break

        print(f"Processing: {pdf.name}")

        out_file = OUT_DIR / f"{pdf.stem}.json"
        if out_file.exists():
            continue

        try:
            tei_xml = parse_with_grobid(pdf)
            paper = tei_to_json(tei_xml, pdf.stem)

            # WRITE JSON EVEN IF ABSTRACT IS WEAK
            out_file.write_text(json.dumps(paper, indent=2))
            parsed += 1
            print(f"[OK] {pdf.name}")

        except Exception as e:
            skipped += 1
            print(f"[SKIP] {pdf.name} | {e}")

        time.sleep(2)  # small cooldown to keep GROBID responsive

    print("\n=== arXiv PDF PARSING COMPLETE ===")
    print(f"Parsed : {parsed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    run()
