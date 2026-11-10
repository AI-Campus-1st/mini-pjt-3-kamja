"""chunks 테이블의 청크 564개(common + reason + time)를 KURE-v1로 임베딩해 store/chroma에 적재 (기획서 5-4)
- 컬렉션 하나(kb_kure_v1)에 모두 넣고, 검색할 때 scheme으로 거름: common + reason 또는 common + time
- 다시 실행하면 컬렉션을 지우고 새로 만듦 (같은 청크가 두 번 들어가지 않게)
- 임베딩 생성·인덱스 쓰기는 이 파일(indexer)에서만. app은 조회만
실행 (프로젝트 루트에서, 백그라운드): nohup python -m indexer.embed > report/evidence/embed_log.txt 2>&1 &
"""

import json, sqlite3, time
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL, COLLECTION = "nlpai-lab/KURE-v1", "kb_kure_v1"


def main():
    import chromadb, torch
    from sentence_transformers import SentenceTransformer
    with sqlite3.connect(ROOT / "store" / "docs.db") as con:
        ch = pd.read_sql("SELECT chunk_id, scheme, text, meta FROM chunks", con)
    metas = [{**json.loads(m), "scheme": s, "chunk_id": c} for m, s, c in zip(ch.meta, ch.scheme, ch.chunk_id)]
    print(f"청크 {len(ch)}개:", ch.scheme.value_counts().to_dict(), flush=True)

    t0 = time.time()
    model = SentenceTransformer(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    vecs = model.encode(ch.text.tolist(), batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    print(f"임베딩 {time.time() - t0:.0f}초 (모델 불러오기 포함), 차원 {vecs.shape[1]}", flush=True)

    client = chromadb.PersistentClient(path=str(ROOT / "store" / "chroma"))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "model": MODEL,
                                                         "built_at": datetime.now().isoformat(timespec="seconds")})
    col.add(ids=ch.chunk_id.tolist(), documents=ch.text.tolist(), embeddings=vecs.tolist(), metadatas=metas)
    print(f"적재 완료: {COLLECTION} {col.count()}개 → store/chroma", flush=True)


if __name__ == "__main__":
    main()
