import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()

API_KEY = os.getenv("CORE_API_KEY")
if not API_KEY:
    raise RuntimeError("CORE_API_KEY not found in .env")

# --------------------------------------------------
# Paths
# --------------------------------------------------
RAW_DIR = Path("data/raw/core")
META_DIR = RAW_DIR / "metadata"

RAW_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# API config
# --------------------------------------------------
URL = "https://api.core.ac.uk/v3/search/works"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "User-Agent": "research-intelligence/1.0"
}

# --------------------------------------------------
# Downloader
# --------------------------------------------------
def download_core(limit=1000):
    downloaded = 0
    offset = 0

    while downloaded < limit:
        params = {
            "q": "machine learning",
            "limit": 100,
            "offset": offset
        }

        # -------- API CALL (guarded) --------
        try:
            resp = requests.get(
                URL,
                headers=HEADERS,
                params=params,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

        except Exception as e:
            print(f"[CORE API ERROR] offset={offset} | {e}")
            print("Stopping CORE ingestion safely.")
            break

        results = data.get("results", [])
        if not results:
            print("No more CORE results.")
            break

        # -------- DOWNLOAD PDFs --------
        for item in results:
            if downloaded >= limit:
                break

            paper_id = item.get("id")
            pdf_url = item.get("downloadUrl")

            if not paper_id or not pdf_url:
                continue

            pdf_path = RAW_DIR / f"{paper_id}.pdf"
            meta_path = META_DIR / f"{paper_id}.json"

            # Skip if already downloaded and valid
            if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
                continue

            try:
                r = requests.get(
                    pdf_url,
                    timeout=60,
                    stream=True,
                    headers={"User-Agent": "research-intelligence/1.0"}
                )
                r.raise_for_status()

                with open(pdf_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            except Exception as e:
                print(f"[SKIP] CORE PDF {paper_id} | {e}")
                continue

            # -------- Metadata --------
            meta = {
                "id": paper_id,
                "title": item.get("title"),
                "year": item.get("yearPublished"),
                "source": "core"
            }

            meta_path.write_text(json.dumps(meta, indent=2))

            downloaded += 1
            print(f"[OK] CORE {paper_id} ({downloaded}/{limit})")

            time.sleep(1)  # politeness

        offset += 100

    print(f"CORE ingestion finished: {downloaded} PDFs downloaded")
