"""선별 뉴스 16건의 본문을 제공 묶음 CSV에서 원문URL로 찾아 docs.db의 news_provided 테이블에 원문 그대로 저장
- 정제(고지문·사진 설명 제거)는 indexer에서 한다 (collector는 원문 그대로)
- 묶음 파일 안 URL 중복 4건 대비: 같은 URL이 여러 행이면 첫 행만 유지 (기획서 3-3)
실행 (프로젝트 루트에서): python -m collector.load_provided
"""
import sqlite3, unicodedata
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "store" / "docs.db"
BUNDLE_PREFIX = "02_AI반도체_묶음"


def main():
    sel = pd.read_csv(ROOT / "collector" / "news_selected.csv", encoding="utf-8-sig")
    bundle = next(p for p in (ROOT / "data" / "provided").iterdir()
                  if unicodedata.normalize("NFC", p.name).startswith(BUNDLE_PREFIX))
    urls = set(sel["원문URL"])
    chunks = pd.read_csv(bundle, encoding="utf-8-sig", chunksize=20000)   # 파일이 커서 나눠 읽기
    hits = pd.concat(c[c["원문URL"].isin(urls)] for c in chunks)
    dup = hits["원문URL"].duplicated(keep="first")
    if dup.any():
        print(f"URL 중복 {dup.sum()}건 → 첫 행만 유지")
    df = sel[["doc_id", "tier", "원문URL", "본문길이"]].merge(hits[~dup], on="원문URL", how="left")
    lost = df["본문전문"].isna()
    if lost.any():
        raise ValueError(f"본문을 못 찾은 기사: {df.loc[lost, 'doc_id'].tolist()}")

    diff = df[df["본문전문"].str.len() != df["본문길이"]]   # 선별 때 잰 길이와 비교 (같은 기사인지 확인)
    print(f"본문 길이 불일치 {len(diff)}건", diff[["doc_id", "본문길이"]].values.tolist() if len(diff) else "")
    df = df.rename(columns={"제목": "title", "본문전문": "text", "원문URL": "url", "발행일": "published_at", "매체명": "source"})
    df["collected_at"] = datetime.now().isoformat(timespec="seconds")
    df = df[["doc_id", "tier", "title", "source", "url", "published_at", "collected_at", "text"]]
    with sqlite3.connect(DB) as con:
        df.to_sql("news_provided", con, if_exists="replace", index=False)
    print(f"news_provided {len(df)}건 저장, 본문 평균 {df.text.str.len().mean():.0f}자")


if __name__ == "__main__":
    main()
