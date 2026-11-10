"""본 적재 전 임베딩 모델 비교: KURE-v1 vs text-embedding-3-large (기획서 5-4)
- 같은 청크(common + 댓글 이유별 안)를 메모리에서만 임베딩 → store/chroma에는 쓰지 않음
- 평가셋 32문항(답변 불가 제외)으로 Hit@1·Hit@3·MRR@10, 소요 시간(모델 불러오기 포함), OpenAI 토큰 수·비용
- 정답: 정부 청크는 쪽 범위에 gold_pages가 들면, 사례 청크는 case_id가 gold_case_ids에 있으면
실행 (프로젝트 루트에서): python -m eval.compare_embed
"""

import json, os, sqlite3, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results"
PRICE = 0.13   # 100만 토큰당 $ (text-embedding-3-large, 기획서 5-4·12-1)


def load():
    with sqlite3.connect(ROOT / "store" / "docs.db") as con:
        ch = pd.read_sql("SELECT chunk_id, text, meta FROM chunks WHERE scheme IN ('common', 'reason')", con)
    ch["meta"] = ch.meta.map(json.loads)
    qs = [json.loads(l) for l in open(ROOT / "eval" / "eval_set.jsonl", encoding="utf-8")]
    return ch, [q for q in qs if q["type"] != "unanswerable"]


def relevant(m, q):
    if m["doc_type"] == "gov":
        pages = q.get("gold_pages", {}).get(m["doc_id"], [])
        return any(m["page_start"] <= p <= m["page_end"] for p in pages)
    return m.get("case_id") in q.get("gold_case_ids", []) and m["doc_type"] in ("case_summary", "youtube_comment")


def kure(texts, queries):
    import torch
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer("nlpai-lab/KURE-v1", device="mps" if torch.backends.mps.is_available() else "cpu")
    t_load = time.time() - t0
    enc = lambda x: model.encode(x, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    t1 = time.time(); D = enc(texts); t_doc = time.time() - t1
    t2 = time.time(); Q = enc(queries); t_q = time.time() - t2
    return D, Q, {"load_s": round(t_load, 1), "doc_s": round(t_doc, 1), "query_s": round(t_q, 2), "tokens": None, "cost_usd": 0}


def openai_large(texts, queries):
    import tiktoken
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI(default_headers={"Accept-Encoding": "gzip, deflate"})   # 키는 .env의 OPENAI_API_KEY
    tokens = sum(len(tiktoken.get_encoding("cl100k_base").encode(t)) for t in texts + queries)

    def enc(x):
        out = []
        for i in range(0, len(x), 100):
            r = client.embeddings.create(model="text-embedding-3-large", input=x[i:i + 100])
            out += [d.embedding for d in r.data]
        return np.array(out)
    t1 = time.time(); D = enc(texts); t_doc = time.time() - t1
    t2 = time.time(); Q = enc(queries); t_q = time.time() - t2
    return D, Q, {"load_s": 0, "doc_s": round(t_doc, 1), "query_s": round(t_q, 2), "tokens": tokens,
                  "cost_usd": round(tokens / 1e6 * PRICE, 4)}


def score(D, Q, ch, qs, tag):
    rows = []
    for q, qv in zip(qs, Q):
        top = np.argsort(-(D @ qv))[:10]
        hits = [relevant(ch.meta.iloc[i], q) for i in top]
        rank = hits.index(True) + 1 if any(hits) else None
        rows.append({"run_tag": tag, "id": q["id"], "type": q["type"], "hit1": int(hits[0]), "hit3": int(any(hits[:3])),
                     "rr": round(1 / rank, 3) if rank else 0, "top3": " | ".join(ch.chunk_id.iloc[i] for i in top[:3])})
    return pd.DataFrame(rows)


def main():
    ch, qs = load()
    texts, queries = ch.text.tolist(), [q["q"] for q in qs]
    print(f"청크 {len(texts)}개, 문항 {len(qs)}개")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for tag, fn in [("exp0_kure_v1", kure), ("exp0_openai_large", openai_large)]:
        D, Q, cost = fn(texts, queries)
        df = score(D, Q, ch, qs, tag)
        df.to_csv(OUT / f"{tag}.csv", index=False, encoding="utf-8-sig")
        summary.append({"run_tag": tag, "hit1": round(df.hit1.mean(), 3), "hit3": round(df.hit3.mean(), 3),
                        "mrr": round(df.rr.mean(), 3), **cost})
        print(f"\n[{tag}] 유형별 Hit@3\n" + df.groupby("type").hit3.mean().round(2).to_string())
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "exp0_embed_compare.csv", index=False, encoding="utf-8-sig")
    print("\n" + s.to_string(index=False))


if __name__ == "__main__":
    main()
