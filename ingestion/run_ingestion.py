from ingestion.arxiv import download_arxiv
from ingestion.pmc import download_pmc
from ingestion.core import download_core

def run():
    print("Starting raw data ingestion")

    print("Downloading arXiv papers...")
    download_arxiv(limit=2000)

    print("Downloading PMC papers...")
    download_pmc(target_count=2000)

    print("Downloading CORE papers...")
    download_core(limit=1000)

    print("Raw data ingestion complete")

if __name__ == "__main__":
    run()
