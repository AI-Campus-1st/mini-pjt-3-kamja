"""
더미 기획서 정답표 검사 (심화 9-3, 구현 전에 1회). LLM 없음, store는 읽기만
- 문장이 기획서 원문에 그대로 있는지, anchor가 기획서에서 한 번만 나오고 문장 안에 있는지
- 특징 값이 features.ALLOWED 안인지, period 형식
- 정답 사례·등급을 reaction.predict(정답 특징)로 다시 계산해 정답표와 같은지
실행 (프로젝트 루트): python -m eval.check_plan_gold [--gold eval/plans/plan_gold2.jsonl]
"""

import argparse, json, re, sys, unicodedata
from pathlib import Path

from rag.features import ALLOWED, PERIOD
from rag.reaction import predict

ROOT = Path(__file__).resolve().parents[1]


def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()   # 맥 한글 NFC + 공백 정리


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/plans/plan_gold.jsonl")
    gold = [json.loads(l) for l in (ROOT / ap.parse_args().gold).read_text(encoding="utf-8").splitlines() if l.strip()]
    plans = {p: norm((ROOT / "eval" / "plans" / f"plan_{p}.md").read_text(encoding="utf-8")) for p in {g["plan_id"] for g in gold}}
    bad = 0
    for g in gold:
        plan, s, a, ft = plans[g["plan_id"]], norm(g["sentence"]), norm(g["anchor"]), g["features"]
        errs = [] if s in plan else ["문장이 원문에 없음"]
        if plan.count(a) != 1 or a not in s:
            errs.append(f"anchor {plan.count(a)}회·문장 안 {a in s}")
        errs += [f"{k} 허용값 밖: {v}" for k, v in ft.items() if v is not None and k != "period" and v not in ALLOWED[k]]
        if ft["period"] is not None and not PERIOD.match(ft["period"]):
            errs.append(f"period 형식: {ft['period']}")
        if g["is_ai"]:
            r = predict(ft)
            ids = sorted(x["case_id"] for x in r["similar"])
            if ids != sorted(g["gold_case_ids"]):
                errs.append(f"사례 {ids}")
            if r["grade"] != g["gold_grade"]:
                errs.append(f"등급 {r['grade']}")
        bad += bool(errs)
        print(g["item_id"], "OK" if not errs else "X " + " | ".join(errs))
    print(f"\n{len(gold)}개 중 문제 {bad}개 (AI 항목 {sum(g['is_ai'] for g in gold)}개)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
