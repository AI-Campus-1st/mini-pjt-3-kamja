"""
평가셋으로 검색·예상 성능 측정 (기획서 6-1). 답변 불가 8문항은 Hit에서 제외(32문항), 의도 판정에는 포함
- mode chain (기본): rag/chain.py 경유 (챗봇과 같은 경로) → Hit + 의도·특징 추출·등급 일치
- mode search: 검색기만 (exp2 재현용, --scheme·--qualified)
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m eval.run_eval --tag run1_chain
결과: eval/results/<tag>.json (요약), <tag>_detail.csv (문항별)
"""

import argparse, json
from pathlib import Path

import pandas as pd
from rag.retriever import search

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["industry", "ai_part", "disclosed", "period"]


def is_hit(meta, item):
    """정부 청크: 쪽 범위에 gold_pages가 들면 정답 / 사례 요약·댓글: case_id가 gold_case_ids에 있으면 정답"""
    pages = item.get("gold_pages", {})
    if meta.get("doc_type") == "gov":
        return meta.get("doc_id") in pages and any(meta["page_start"] <= p <= meta["page_end"] for p in pages[meta["doc_id"]])
    return meta.get("case_id") in item.get("gold_case_ids", [])


def run_one(x, a):
    if a.mode == "search":
        res = search(x["q"], k=10, scheme=a.scheme, qualified_only=a.qualified and x["type"] == "predict")
        return res, {}
    from rag.chain import retrieve
    r = retrieve(x["q"], k=10)
    p = r["prediction"]
    return r["docs"], {"intent": r["intent"], "features": json.dumps(r["features"], ensure_ascii=False),
                       "grade": p["grade"] if p else None, "feat_ok": all(r["features"][f] == x.get("features", {}).get(f) for f in FEATS),
                       **{f"f_{f}": r["features"][f] == x.get("features", {}).get(f) for f in FEATS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="chain", choices=["chain", "search"])
    ap.add_argument("--scheme", default="time", choices=["reason", "time"])
    ap.add_argument("--qualified", action="store_true", help="search 모드: 예상 문항만 탈락 사례 제외 (유형 정답 기준)")
    a = ap.parse_args()

    lines = (ROOT / "eval" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()
    items = [json.loads(l) for l in lines if l.strip()]
    if a.mode == "search":
        items = [x for x in items if x["type"] != "unanswerable"]

    rows = []
    for x in items:
        res, extra = run_one(x, a)
        hits = [is_hit(r["meta"], x) for r in res]
        rank = hits.index(True) + 1 if True in hits else None
        rows.append({"id": x["id"], "type": x["type"], "rank": rank, "hit1": int(rank == 1),
                     "hit3": int(rank is not None and rank <= 3), "rr": 1 / rank if rank else 0.0,
                     "top3": " | ".join(f'{r["id"]}({r["score"]})' for r in res[:3]), **extra})
    df = pd.DataFrame(rows)
    h = df[df.type != "unanswerable"]

    summary = {"tag": a.tag, "mode": a.mode, "scheme": a.scheme if a.mode == "search" else "time", "qualified": a.qualified,
               "n": len(h), "hit1": round(h.hit1.mean(), 3), "hit3": round(h.hit3.mean(), 3), "mrr10": round(h.rr.mean(), 3),
               "by_type": h.groupby("type")[["hit1", "hit3", "rr"]].mean().round(3).to_dict("index")}
    if a.mode == "chain":
        p = df[df.type == "predict"]
        o = df[df.type.isin(["fact_reaction", "fact_policy", "compare", "unanswerable"])]   # 종합은 예상이 섞여 있어 의도 채점에서 뺌
        exp = {x["id"]: x.get("expected_grade") for x in items}
        summary.update({"intent_predict_recall": round((p.intent == "predict").mean(), 3),
                        "intent_other_ok": round((o.intent == "other").mean(), 3),
                        "feat_all_ok": round(p.feat_ok.mean(), 3),
                        "feat_by_field": {f: round(p[f"f_{f}"].mean(), 3) for f in FEATS},
                        "grade_ok": round((p.grade == p.id.map(exp)).mean(), 3),
                        "synthesis_intent": df[df.type == "synthesis"].set_index("id").intent.to_dict()})

    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out / f"{a.tag}_detail.csv", index=False, encoding="utf-8-sig")

    print(f'[{a.tag}] n={summary["n"]}  Hit@1 {summary["hit1"]}  Hit@3 {summary["hit3"]}  MRR {summary["mrr10"]}')
    print(h.groupby("type")[["hit1", "hit3", "rr"]].mean().round(3).to_string())
    if a.mode == "chain":
        print({k: summary[k] for k in ["intent_predict_recall", "intent_other_ok", "feat_all_ok", "feat_by_field", "grade_ok", "synthesis_intent"]})
        bad = df[(df.type == "predict") & ((df.feat_ok == False) | (df.grade != df.id.map(exp)))]
        print("\n[예상 문항: 특징·등급 틀림]\n" + bad[["id", "intent", "features", "grade"]].to_string(index=False))
        wrong = o[o.intent != "other"]
        print("\n[예상이 아닌데 predict로 본 문항]\n" + (wrong[["id", "type"]].to_string(index=False) if len(wrong) else "없음"))
    print("\n[Hit@3 실패 문항]\n" + h[h.hit3 == 0][["id", "type", "rank", "top3"]].to_string(index=False))


if __name__ == "__main__":
    main()
