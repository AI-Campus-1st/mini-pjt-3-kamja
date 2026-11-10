"""
심화 9-3 측정 (설계: report/evidence/plan_check_design.md, 결과 전 고정). 정답표 eval/plans/plan_gold.jsonl
① 항목 추출 재현율 ② 특징 추출 정확도 ③ 사례 검색 Hit@3 — A(항목 문장, 범위 그대로) vs B(추출 특징으로 범위 좁힘)
- A·참고값(기획서 전체 검색)은 LLM이 없어 1회, ①②③B는 --n회 반복
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m eval.plan_eval --tag deep_plan [--gold eval/plans/plan_gold2.jsonl]
결과: eval/results/<tag>_A·_full·_r1~rN (원자료 _detail.csv 먼저), <tag>_summary.json
"""

import argparse, json, re, unicodedata
from pathlib import Path

import pandas as pd

from rag.features import extract
from rag.plan_check import base_docs, extract_items, item_docs
from rag.reaction import predict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results"
FEATS = ["industry", "ai_part", "disclosed", "period"]


def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()


def rank_of(docs, gold_ids):
    """run_eval.is_hit과 같은 기준: case_id가 정답 사례면 맞음. 몇 번째인지 (없으면 None)"""
    hits = [d["meta"].get("case_id") in gold_ids for d in docs]
    return hits.index(True) + 1 if True in hits else None


def score(rank):
    return {"rank": rank, "hit3": int(rank is not None and rank <= 3), "rr": 1 / rank if rank else 0.0}


def save(df, tag, summary):
    df.to_csv(OUT / f"{tag}_detail.csv", index=False, encoding="utf-8-sig")   # 집계 전에 원자료 저장
    (OUT / f"{tag}.json").write_text(json.dumps({"tag": tag, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")


def run_search(tag, gold, plans, raw):
    """A: 항목 문장 그대로 / full(참고): 기획서 전체 한 번 검색으로 그 기획서의 항목을 모두 채점"""
    hit_items = [g for g in gold if g["is_ai"] and g["gold_case_ids"]]
    a = pd.DataFrame([{"item_id": g["item_id"], **score(rank_of(docs := base_docs(g["sentence"]), g["gold_case_ids"])),
                       "top3": " | ".join(d["id"] for d in docs[:3])} for g in hit_items])
    save(a, f"{tag}_A", {"n": len(a), "hit3": round(a.hit3.mean(), 3), "mrr": round(a.rr.mean(), 3)})
    full_docs = {p: base_docs(raw[p]) for p in plans}
    f = pd.DataFrame([{"item_id": g["item_id"], **score(rank_of(full_docs[g["plan_id"]], g["gold_case_ids"])),
                       "top3": " | ".join(d["id"] for d in full_docs[g["plan_id"]][:3])} for g in hit_items])
    save(f, f"{tag}_full", {"n": len(f), "hit3": round(f.hit3.mean(), 3), "mrr": round(f.rr.mean(), 3), "note": "참고값, 결론에 쓰지 않음"})
    return a, f


def match(quote, plan, cands, used):
    """인용이 원문에 그대로 있고 아직 안 쓴 정답 항목의 anchor를 포함하면 그 항목 (추출 하나 = 정답 하나)"""
    q = norm(quote)
    if not q or q not in plan:
        return "invalid"
    return next((g for g in cands if g["item_id"] not in used and norm(g["anchor"]) in q), None)


def run_once(tag, run, gold, plans, raw):
    ext_rows, item_rows = [], []
    for p in plans:
        cands, used, found = [g for g in gold if g["plan_id"] == p], set(), {}
        for it in extract_items(raw[p]):
            m = match(it["quote"], plans[p], cands, used)
            kind = "invalid" if m == "invalid" else "none" if m is None else "ai" if m["is_ai"] else "decoy"
            if kind in ("ai", "decoy"):
                used.add(m["item_id"])
            if kind == "ai":
                found[m["item_id"]] = it
            ext_rows.append({"run": run, "plan_id": p, "kind": kind, "matched": m["item_id"] if isinstance(m, dict) else None,
                             "quote": it["quote"], "question": it["question"], "common": it.get("common")})
        for g in (g for g in cands if g["is_ai"]):
            row = {"run": run, "item_id": g["item_id"], "found": g["item_id"] in found, "has_gold": bool(g["gold_case_ids"])}
            if row["found"]:
                q = found[g["item_id"]]["question"]
                f = extract(q)["features"]
                pred = predict(f)
                row.update({"question": q, "features": json.dumps(f, ensure_ascii=False, sort_keys=True),
                            **{f"ok_{k}": f[k] == g["features"][k] for k in FEATS},
                            "feat_ok": all(f[k] == g["features"][k] for k in FEATS),
                            "grade": pred["grade"], "grade_ok": pred["grade"] == g["gold_grade"],
                            "similar_ok": sorted(s["case_id"] for s in pred["similar"]) == sorted(g["gold_case_ids"])})
            if g["gold_case_ids"]:   # ③ B: 검색어는 정답표 문장으로 고정, 범위만 추출 특징으로
                if row["found"]:
                    docs, _ = item_docs(g["sentence"], f)
                    row.update(score(rank_of(docs, g["gold_case_ids"])), top3=" | ".join(d["id"] for d in docs[:3]))
                else:
                    row.update(score(None))   # 항목을 못 찾으면 예상이 나오지 않으므로 0점
            item_rows.append(row)
    ext, items = pd.DataFrame(ext_rows), pd.DataFrame(item_rows)
    ext.to_csv(OUT / f"{tag}_r{run}_extract.csv", index=False, encoding="utf-8-sig")
    fd = items[items.found]
    hb = items[items.has_gold]   # ③은 정답 사례가 있는 항목만
    s = {"recall": round(items.found.mean(), 3), "n_ai": len(items), "n_found": int(items.found.sum()),
         "wrong": int((ext.kind != "ai").sum()), "wrong_by_kind": ext[ext.kind != "ai"].kind.value_counts().to_dict(),
         "field_acc": round(fd[[f"ok_{k}" for k in FEATS]].to_numpy().mean(), 3) if len(fd) else None,
         "field_acc_by_feat": {k: round(fd[f"ok_{k}"].mean(), 3) for k in FEATS} if len(fd) else {},
         "feat_ok": round(fd.feat_ok.mean(), 3) if len(fd) else None,
         "grade_ok": round(fd.grade_ok.mean(), 3) if len(fd) else None,
         "similar_ok": round(fd.similar_ok.mean(), 3) if len(fd) else None,
         "B_n": len(hb), "B_hit3": round(hb.hit3.mean(), 3), "B_mrr": round(hb.rr.mean(), 3),
         "B_unfound": int((~hb.found).sum())}
    save(items, f"{tag}_r{run}", s)
    print(f"run {run}: 재현율 {s['recall']} · 잘못 찾음 {s['wrong']} · 칸 {s['field_acc']} · 4개 모두 {s['feat_ok']} · B Hit@3 {s['B_hit3']}", flush=True)
    return s, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="deep_plan")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--gold", default="eval/plans/plan_gold.jsonl")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gold = [json.loads(l) for l in (ROOT / a.gold).read_text(encoding="utf-8").splitlines() if l.strip()]
    pids = sorted({g["plan_id"] for g in gold})
    raw = {p: unicodedata.normalize("NFC", (ROOT / "eval" / "plans" / f"plan_{p}.md").read_text(encoding="utf-8")) for p in pids}
    plans = {p: norm(raw[p]) for p in pids}

    A, full = run_search(a.tag, gold, plans, raw)
    print(f"A Hit@3 {A.hit3.mean():.3f} · 참고(기획서 전체) Hit@3 {full.hit3.mean():.3f}", flush=True)
    runs, all_items = [], []
    for r in range(1, a.n + 1):
        s, items = run_once(a.tag, r, gold, plans, raw)
        runs.append(s)
        all_items.append(items)

    df = pd.concat(all_items)
    fv = df[df.found].assign(v=lambda d: d.features.fillna("(없음)")).groupby("item_id").v.nunique()   # 빈 값도 하나의 값
    a_hit = round(A.hit3.mean(), 3)
    b = [x["B_hit3"] for x in runs]
    verdict = "B가 낫다" if all(x > a_hit for x in b) else "A가 낫다" if all(x < a_hit for x in b) else "차이 없음"
    keys = ["recall", "wrong", "field_acc", "feat_ok", "grade_ok", "similar_ok", "B_hit3", "B_mrr", "B_unfound"]
    summary = {"n_runs": a.n, **{f"{k}_by_run": [x[k] for x in runs] for k in keys},
               **{f"{k}_mean": round(sum(x[k] for x in runs if x[k] is not None) / max(1, sum(x[k] is not None for x in runs)), 3)
                  for k in keys},
               "A_hit3": a_hit, "A_mrr": round(A.rr.mean(), 3), "full_hit3_ref": round(full.hit3.mean(), 3),
               "verdict_B_vs_A": verdict, "feat_consistency": round((fv == 1).mean(), 3) if len(fv) else None,
               "feat_unstable": sorted(fv[fv > 1].index)}
    (OUT / f"{a.tag}_summary.json").write_text(json.dumps({"tag": f"{a.tag}_summary", "gold": a.gold, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()