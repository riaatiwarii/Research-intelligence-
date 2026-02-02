import arxiv
import json
from pathlib import Path

RAW_DIR = Path("data/raw/arxiv")
META_DIR = RAW_DIR / "metadata"

RAW_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(exist_ok=True)

QUERY = "cat:cs.CL OR cat:cs.LG OR cat:cs.AI"

def download_arxiv(limit=2000):
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=3
    )

    search = arxiv.Search(
        query=QUERY,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for paper in client.results(search):
        paper_id = paper.entry_id.split("/")[-1]
        pdf_path = RAW_DIR / f"{paper_id}.pdf"
        meta_path = META_DIR / f"{paper_id}.json"

        if pdf_path.exists():
            continue

        try:
            paper.download_pdf(dirpath=RAW_DIR, filename=pdf_path.name)
        except Exception as e:
            print(f"[SKIP] {paper_id} | PDF unavailable ({e})")
            continue


        metadata = {
            "id": paper_id,
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary,
            "published": paper.published.isoformat(),
            "categories": paper.categories,
            "source": "arxiv"
        }

        meta_path.write_text(json.dumps(metadata, indent=2))
