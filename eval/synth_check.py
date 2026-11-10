"""
exp10 확인: 종합 판정(코드 규칙)과 답변용 검색 5개의 출처 종류 섞임을 평가셋 S + 확인 질문(W·N)으로 잼 (개선 전·후 같은 스크립트)
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m eval.synth_check --tag exp10_check_before
결과: eval/results/<tag>_detail.csv (원자료 먼저), <tag>.json
"""

import argparse, json
from pathlib import Path

import pandas as pd
from rag.chain import ANSWER_K, retrieve

ROOT = Path(__file__).resolve().parents[1]
KIND = {"gov": "정부", "news_provided": "뉴스"}   # 나머지(사례 요약·댓글)는 반응


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ev = [json.loads(l) for l in (ROOT / "eval" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pr = [json.loads(l) for l in (ROOT / "eval" / "probes" / "synth_probe.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [{"id": x["id"], "q": x["q"], "synth": True} for x in ev if x["type"] == "synthesis"] + pr

    rows = []
    for x in items:
        r = retrieve(x["q"], k=ANSWER_K)
        kinds = [KIND.get(d["meta"].get("doc_type"), "반응") for d in r["docs"]]
        rows.append({"id": x["id"], "gold_synth": x["synth"], "synth": r.get("synthesis"), "intent": r["intent"],
                     "kinds": " ".join(kinds), "both_top3": "반응" in kinds[:3] and "정부" in kinds[:3],
                     "both_top5": "반응" in kinds and "정부" in kinds})
    df = pd.DataFrame(rows)
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.tag}_detail.csv", index=False, encoding="utf-8-sig")   # 원자료 먼저

    s_other = df[df.id.str.startswith("S") & (df.intent != "predict")]   # 예상 경로(S04 등)는 대상 밖
    w = df[df.id.str.startswith("W")]
    s = {"tag": a.tag,
         "S_nonpredict": list(s_other.id), "S_both_top3": int(s_other.both_top3.sum()), "S_both_top5": int(s_other.both_top5.sum()),
         "W_both_top3": int(w.both_top3.sum()), "W_n": len(w),
         "detect_ok": int((df.synth.fillna(False) == df.gold_synth).sum()), "detect_n": len(df)}   # 개선 전은 synth가 없음(None)
    (out / f"{a.tag}.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(df[["id", "gold_synth", "synth", "intent", "kinds", "both_top3"]].to_string(index=False))


if __name__ == "__main__":
    main()
