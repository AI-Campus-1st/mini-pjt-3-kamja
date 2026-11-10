"""정부 PDF에서 쪽별 글자를 뽑아 docs.db의 gov_pages 테이블에 저장 (pypdf)
- 탭 → 공백 (2026 안내서), 쪽 번호만 있는 줄 제거
- 여러 쪽(REPEAT 비율 이상)에 똑같이 나오는 줄은 머리글·옆면 목차로 보고 제거 → 제거한 줄을 출력해 눈으로 확인
- page = PDF 쪽 순서(1부터). 샘플 확인 결과 인쇄된 쪽 번호와 같음 (2023 11쪽, 2026 8쪽)
실행 (프로젝트 루트에서): python -m indexer.parse
"""
import re, sqlite3
from collections import Counter
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "store" / "docs.db"
REPEAT = 0.3


def lines_of(text):
    return [re.sub(r" {2,}", " ", l.replace("\t", " ")).strip() for l in text.splitlines()]


def main():
    with sqlite3.connect(DB) as con:
        docs = pd.read_sql("SELECT doc_id, file_path FROM gov_docs", con)
    rows = []
    for d in docs.itertuples():
        pages = [lines_of(p.extract_text() or "") for p in PdfReader(ROOT / d.file_path).pages]
        freq = Counter(l for ls in pages for l in set(ls) if l)
        header = {l for l, n in freq.items() if n >= len(pages) * REPEAT}
        for i, ls in enumerate(pages, 1):
            text = "\n".join(l for l in ls if l and l not in header and not l.isdigit())
            rows.append({"doc_id": d.doc_id, "page": i, "text": text, "n_chars": len(text)})
        empty = sum(1 for r in rows if r["doc_id"] == d.doc_id and r["n_chars"] < 50)
        print(f"{d.doc_id}: {len(pages)}쪽, 글자 50자 미만 쪽 {empty}개, 머리글로 제거한 줄 {len(header)}종")
        for h in sorted(header):
            print("   -", h[:60])
    with sqlite3.connect(DB) as con:
        pd.DataFrame(rows).to_sql("gov_pages", con, if_exists="replace", index=False)
    print(f"gov_pages {len(rows)}쪽 저장")


if __name__ == "__main__":
    main()
