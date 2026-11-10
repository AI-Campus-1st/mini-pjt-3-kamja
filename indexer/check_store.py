"""store/chroma 적재 검증: 청크 수 대조, 메타데이터 결손, 샘플 검색 (기획서 10항 Day 3 오전 5번)
실행 (프로젝트 루트에서): python -m indexer.check_store
"""

import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "kb_kure_v1"
COMMON = ["doc_id", "title", "source", "url", "published_at", "collected_at", "doc_type", "authority", "chunk_index", "scheme"]
NEED = {"youtube_comment": ["case_id", "industry", "ai_part", "disclosed", "period", "video_kind", "qualified"],
        "case_summary": ["case_id", "industry", "ai_part", "disclosed", "period", "qualified"],
        "gov": ["section", "page_start", "page_end"], "news_provided": []}
SAMPLES = [("reason", "이니스프리 AI 모델 광고에 소비자들은 어떻게 반응했어?"),
           ("reason", "AI로 만든 광고 이미지도 저작권 등록을 할 수 있나?"),
           ("time", "AI 광고가 창작자 일자리를 뺏는다는 반응이 있었어?")]


def main():
    import chromadb, torch
    from sentence_transformers import SentenceTransformer
    col = chromadb.PersistentClient(path=str(ROOT / "store" / "chroma")).get_collection(COLLECTION)
    got = col.get(include=["metadatas"])
    m = pd.DataFrame(got["metadatas"])
    with sqlite3.connect(ROOT / "store" / "docs.db") as con:
        n_db = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"① 청크 수: chroma {col.count()} / docs.db chunks {n_db} → {'일치' if col.count() == n_db else '불일치!'}")
    print(m.groupby(["doc_type", "scheme"]).size().to_string())

    print("\n② 메타데이터 결손 (문서 종류별 빠진 개수, 0이면 생략)")
    for t, g in m.groupby("doc_type"):
        miss = {f: int(g[f].isna().sum()) if f in g else len(g) for f in COMMON + NEED[t]}
        miss = {f: n for f, n in miss.items() if n}
        print(f"  {t} ({len(g)}개): {miss or '결손 없음'}")

    print("\n③ 샘플 검색 (상위 3개)")
    model = SentenceTransformer(col.metadata["model"], device="mps" if torch.backends.mps.is_available() else "cpu")
    for scheme, q in SAMPLES:
        v = model.encode([q], normalize_embeddings=True).tolist()
        r = col.query(query_embeddings=v, n_results=3, where={"scheme": {"$in": ["common", scheme]}})
        print(f"  [{scheme}] {q}")
        for cid, dist, doc in zip(r["ids"][0], r["distances"][0], r["documents"][0]):
            print(f"    {1 - dist:.3f} {cid} | {doc.splitlines()[0][:50]}")


if __name__ == "__main__":
    main()
