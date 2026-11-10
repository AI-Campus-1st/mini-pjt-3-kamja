"""
LLM 판정 3회 반복 일치율 (기획서 11항, Day 5): 범위(check_scope)·특징/의도(extract)·비교(detect)를 같은 질문에 3번씩
- 대상: 평가셋 40문항 + 범위 확인 질문 3개("수치 질문 + 광고 특징" 형태, 프롬프트 예시와 다른 값)
- 일치율: 3번 결과가 모두 같은 문항 비율 / 정답률: 실행마다 정답과 같은 비율의 평균
- 범위 정답: 답변 불가는 거절(U02는 legal 약한 거절), 나머지는 강한 거절이 아니면 정답
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 nohup python -m eval.repeat3 --tag day5_repeat3_v1 > report/evidence/day5_repeat3_v1.txt 2>&1 &
"""

import argparse, json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from rag.compare import detect
from rag.features import extract
from rag.reaction import FEATS
from rag.scope import check_scope

ROOT = Path(__file__).resolve().parents[1]
PROBES = [   # 수치 질문에 광고 특징을 붙인 형태 (기대: numbers 거절)
    {"id": "X01", "type": "probe", "q": "화장품 브랜드가 AI 모델 인물로 광고를 만들고 처음부터 공개하면 매출이 몇 % 줄어들까?"},
    {"id": "X02", "type": "probe", "q": "공공기관이 홍보 영상 전체를 AI로 만들어 표기 없이 올리면 구독자가 몇 명 줄지 예상해줘"},
    {"id": "X03", "type": "probe", "q": "교육 브랜드가 2025년에 AI 숏폼을 처음부터 AI라고 밝히고 내면 클릭률이 몇 % 오를까?"},
]


def scope_ok(x, s):
    hard = s["scope"] and not s["soft"]
    if x["id"] == "U02":
        return s["scope"] == "legal"
    if x["type"] in ("unanswerable", "probe"):
        return bool(hard)
    return not hard


def one(x):
    s, e, c = check_scope(x["q"]), extract(x["q"]), detect(x["q"])
    gold = {f: x.get("features", {}).get(f) for f in FEATS}
    return {"scope": s["scope"], "scope_ok": scope_ok(x, s), "intent": e["intent"],
            "features": json.dumps(e["features"], ensure_ascii=False, sort_keys=True),
            "feat_ok": e["features"] == gold if x["type"] == "predict" else None,
            "compare": json.dumps({k: c[k] for k in ["mode", "case_ids", "by"]}, ensure_ascii=False, sort_keys=True),
            "error": s["error"] or c["error"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    items = [json.loads(l) for l in (ROOT / "eval" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] + PROBES
    rows = []
    for run in range(1, a.n + 1):
        for x in items:
            rows.append({"run": run, "id": x["id"], "type": x["type"], **one(x)})
        print(f"run {run} 완료", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.tag}_detail.csv", index=False, encoding="utf-8-sig")   # 집계 전에 먼저 저장 (집계 오류에도 원자료 보존)

    def stable(col, sub=None):
        d = df if sub is None else df[df.type.isin(sub)]
        g = d.assign(v=d[col].fillna("(없음)")).groupby("id").v.nunique()   # 빈 값(None)도 하나의 값으로 셈
        return round((g == 1).mean(), 3), sorted(g[g > 1].index)

    summary = {"tag": a.tag, "n_runs": a.n, "n_items": len(items)}
    for name, col, sub in [("scope", "scope", None), ("intent", "intent", None), ("features_predict", "features", ["predict"]),
                           ("features_all", "features", None), ("compare_cmp", "compare", ["compare"]), ("compare_all", "compare", None)]:
        r, ids = stable(col, sub)
        summary[name] = {"consistency": r, "unstable_ids": ids}
    summary["scope_ok_mean"] = round(df.scope_ok.mean(), 3)
    summary["scope_ok_probe"] = round(df[df.type == "probe"].scope_ok.mean(), 3)
    summary["scope_ok_unanswerable"] = round(df[df.type == "unanswerable"].scope_ok.mean(), 3)
    summary["wrong_refusal"] = int(((~df.scope_ok) & ~df.type.isin(["unanswerable", "probe"])).sum())
    p = df[df.type == "predict"]
    summary["feat_ok_by_run"] = p.groupby("run").feat_ok.apply(lambda s: round(s.astype(float).mean(), 3)).to_dict()   # 예상 외 문항은 None이라 숫자로 바꿔 계산
    summary["errors"] = int(df.error.notna().sum())

    (out / f"{a.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for i in sorted(set(summary["scope"]["unstable_ids"] + summary["features_predict"]["unstable_ids"] + summary["compare_cmp"]["unstable_ids"])):
        print("\n[" + i + "]\n" + df[df.id == i][["run", "scope", "intent", "features", "compare"]].to_string(index=False))


if __name__ == "__main__":
    main()