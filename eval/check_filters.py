"""필터 확인 (Day 4 오후 5번): ① 필터 없음 = 지금과 같은 검색  ② 필터를 고르면 나온 근거가 모두 조건에 맞는지 (코드 대조)
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m eval.check_filters --tag day4_filters_v1
결과: eval/results/<tag>.json"""
import argparse, json
from pathlib import Path

from rag.chain import retrieve
from rag.filters import allowed_cases, label, normalize
from rag.reaction import cases

ROOT = Path(__file__).resolve().parents[1]
items = {x["id"]: x for x in map(json.loads, open(ROOT / "eval/eval_set.jsonl", encoding="utf-8"))}
IDS = ["P02", "G01", "F01"]
FILTERS = [dict(years=(2025, 2026)), dict(kinds=["정부 기준"]), dict(inds=["식품"]),
           dict(years=(2024, 2026), kinds=["소비자 반응"], inds=["식품"])]


def bad(d, f):
    """조건에 안 맞는 이유 (맞으면 None)"""
    m = d["meta"]
    y = m.get("year")
    if f["lo"] and (y is None or y < f["lo"]) or f["hi"] and (y is None or y > f["hi"]):
        return f"연도 {y}"
    if f["kinds"] and m.get("authority") not in f["kinds"]:
        return f"종류 {m.get('authority')}"
    if f["inds"] and m.get("doc_type") in ("youtube_comment", "case_summary") and m.get("industry") not in f["inds"]:
        return f"업종 {m.get('industry')}"
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
a = ap.parse_args()
log, ok_all = [], True
for i in IDS:
    q = items[i]["q"]
    base, again = retrieve(q, k=5), retrieve(q, k=5, filters=None)
    same = [d["meta"]["doc_id"] for d in base["docs"]] == [d["meta"]["doc_id"] for d in again["docs"]]
    print(f"{i} 필터 없음 = 기본 검색: {same}")
    ok_all &= same
    for raw in FILTERS:
        f = normalize(**raw)
        r = retrieve(q, k=5, filters=f)
        errs = [f"{d['meta']['doc_id']}({bad(d, f)})" for d in r["docs"] if bad(d, f)]
        pred = r["prediction"]
        allow = allowed_cases(f)
        pred_ok = pred is None or all(s["case_id"] in allow for s in pred["similar"])
        ok = not errs and pred_ok
        ok_all &= ok
        print(f"  {label(f)} → 근거 {len(r['docs'])}개 · 조건 밖 {errs or '없음'}"
              + (f" · 비슷한 사례 {[s['case_id'] for s in pred['similar']]} 등급 {pred['grade']}" if pred else ""))
        log.append({"id": i, "filter": label(f), "n_docs": len(r["docs"]), "violations": errs, "pred_ok": pred_ok,
                    "grade": pred["grade"] if pred else None, "pass": ok})
json.dump(log, open(ROOT / f"eval/results/{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n{'모두 통과' if ok_all else '확인 필요'} → eval/results/{a.tag}.json")
