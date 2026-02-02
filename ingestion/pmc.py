import csv
import ftplib
import tarfile
from pathlib import Path

FTP_HOST = "ftp.ncbi.nlm.nih.gov"
BASE_DIR = "/pub/pmc"

RAW_DIR = Path("data/raw/pmc")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FILE_LIST = "oa_file_list.csv"

def download_pmc(target_count=2000):
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login()
    ftp.cwd(BASE_DIR)

    # Download file list if not present
    file_list_path = RAW_DIR / FILE_LIST
    if not file_list_path.exists():
        with open(file_list_path, "wb") as f:
            ftp.retrbinary(f"RETR {FILE_LIST}", f.write)

    extracted = 0

    with open(file_list_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if extracted >= target_count:
                break

            tar_path = row["File"]
            tar_name = tar_path.split("/")[-1]
            local_tar = RAW_DIR / tar_name

            if local_tar.exists():
                continue

            try:
                with open(local_tar, "wb") as f:
                    ftp.retrbinary(f"RETR {tar_path}", f.write)

                with tarfile.open(local_tar, "r:gz") as tar:
                    members = tar.getmembers()
                    tar.extractall(RAW_DIR)
                    extracted += len(members)

                print(f"[OK] PMC extracted {len(members)} XMLs | total ≈ {extracted}")

            except Exception as e:
                print(f"[SKIP] PMC {tar_name} | {e}")
                continue

    ftp.quit()
    print(f"PMC ingestion finished: ~{extracted} XMLs")
