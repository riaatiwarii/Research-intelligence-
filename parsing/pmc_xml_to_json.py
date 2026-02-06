from pathlib import Path
from lxml import etree
import json

RAW_PMC_DIR = Path("data/raw/pmc")
OUT_DIR = Path("data/processed/papers_json")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_pmc_nxml(nxml_path):
    tree = etree.parse(str(nxml_path))
    root = tree.getroot()

    def extract_text(xpath):
        return " ".join(root.xpath(xpath)).strip()

    paper_id = nxml_path.parent.name  # e.g. PMC29085

    paper = {
        "paper_id": paper_id,
        "source": "pmc",
        "title": extract_text(".//article-title//text()"),
        "abstract": extract_text(".//abstract//p//text()"),
        "sections": {}
    }

    for sec in root.xpath(".//sec"):
        title = sec.findtext("title")
        if not title:
            continue

        content = " ".join(sec.xpath(".//p//text()")).strip()
        if len(content) < 100:
            continue

        paper["sections"][title.lower()] = content

    return paper


def run():
    count = 0

    for nxml in RAW_PMC_DIR.rglob("*.nxml"):
        paper = parse_pmc_nxml(nxml)

        if not paper["title"] or not paper["abstract"]:
            continue

        out_file = OUT_DIR / f"{paper['paper_id']}.json"
        out_file.write_text(json.dumps(paper, indent=2))

        count += 1

    print(f"PMC papers extracted: {count}")


if __name__ == "__main__":
    run()
