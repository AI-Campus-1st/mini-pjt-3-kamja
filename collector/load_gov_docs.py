"""정부 안내서 목록(gov_docs.csv)과 PDF 경로를 docs.db의 gov_docs 테이블에 저장 (PDF 원본은 data/gov/에 그대로 둠)
실행 (프로젝트 루트에서): python -m collector.load_gov_docs
"""
import sqlite3, unicodedata
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB, GOV = ROOT / "store" / "docs.db", ROOT / "data" / "gov"
DOWNLOADED = "2026-11-03"   # 수동 다운로드 날짜 (기획서 3-1)


def main():
    df = pd.read_csv(ROOT / "collector" / "gov_docs.csv", encoding="utf-8-sig")
    files = {unicodedata.normalize("NFC", p.name): p for p in GOV.glob("*.pdf")}   # 맥 한글 파일명 대비
    missing = set(df.file_name) - set(files)
    if missing:
        raise FileNotFoundError(f"data/gov에 없는 파일: {missing}")
    df["file_path"] = df.file_name.map(lambda f: str(files[f].relative_to(ROOT)))
    df["collected_at"] = DOWNLOADED
    with sqlite3.connect(DB) as con:
        df.to_sql("gov_docs", con, if_exists="replace", index=False)
    print(f"gov_docs {len(df)}건 저장:", df[["doc_id", "file_name"]].values.tolist())


if __name__ == "__main__":
    main()
