"""
exp9 확인 질문(평가셋 밖 6개, 결과 보기 전 작성)으로 특징 추출을 3회 반복 측정. 개선 전·후 같은 스크립트로 잼
실행 (프로젝트 루트): python -m eval.feat_probe --tag exp9_probe_before
결과: eval/results/<tag>_detail.csv (원자료 먼저), <tag>.json
"""

import argparse, json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from rag.features import extract

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["industry", "ai_part", "disclosed", "period"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    items = [json.loads(l) for l in (ROOT / "eval" / "probes" / "feat_probe.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = []
    for run in range(1, a.n + 1):
        for x in items:
            e = extract(x["q"])
            gold = {f: x["features"].get(f) for f in FEATS}
            rows.append({"run": run, "id": x["id"], "intent": e["intent"], "ai_part": e["features"]["ai_part"],
                         "ai_part_ok": e["features"]["ai_part"] == gold["ai_part"], "feat_ok": e["features"] == gold,
                         "features": json.dumps(e["features"], ensure_ascii=False, sort_keys=True)})
        print(f"run {run} 완료", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.tag}_detail.csv", index=False, encoding="utf-8-sig")   # 집계 전에 원자료 저장

    g = df.assign(v=df.features.fillna("(없음)")).groupby("id").v.nunique()     # 빈 값도 하나의 값으로 셈
    s = {"tag": a.tag, "n_runs": a.n, "n_items": len(items),
         "ai_part_ok_by_run": df.groupby("run").ai_part_ok.mean().round(3).to_dict(),
         "feat_ok_by_run": df.groupby("run").feat_ok.mean().round(3).to_dict(),
         "intent_predict": round((df.intent == "predict").mean(), 3),
         "consistency": round((g == 1).mean(), 3), "unstable_ids": sorted(g[g > 1].index)}
    s["ai_part_ok_mean"] = round(df.ai_part_ok.mean(), 3)
    s["feat_ok_mean"] = round(df.feat_ok.mean(), 3)
    (out / f"{a.tag}.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(df[~df.feat_ok][["run", "id", "features"]].to_string(index=False) if (~df.feat_ok).any() else "모두 맞음")


if __name__ == "__main__":
    main()
