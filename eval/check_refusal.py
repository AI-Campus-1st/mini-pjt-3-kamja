"""
거절 로직 확인 (Day 4): 범위 판정을 평가셋 40문항에 돌려 거절률·잘못 거절률을 재고, 유사도 기준값을 정할 1위 점수 분포를 함께 저장
- 답변 불가 8문항: 범위 밖으로 잡았는지, 유형이 맞는지 (8문항은 Day 2에 같은 유형으로 만들어 낙관적인 수치)
- 나머지 32문항: 잘못 거절한 문항 수 (더 중요한 수치). legal은 약한 거절(답변은 계속)이라 따로 셈
- U02(legal)는 "판단 거절 문구 + 일반 기준 안내"가 되면 거절 성공으로 봄
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m eval.check_refusal --tag day4_scope_v1
결과: eval/results/<tag>.json (요약), <tag>_detail.csv (문항별)
"""

import argparse, json
from pathlib import Path

import pandas as pd
from rag.retriever import search
from rag.scope import SOFT, check_scope

ROOT = Path(__file__).resolve().parents[1]
EXPECT = {"U01": "numbers", "U02": "legal", "U03": "labeling", "U04": "foreign",   # 각 문항 answer_key의 거절 이유
          "U05": "platform", "U06": "audience", "U07": "numbers", "U08": "future"}

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
a = ap.parse_args()

items = [json.loads(l) for l in open(ROOT / "eval/eval_set.jsonl", encoding="utf-8")]
rows = []
for x in items:
    res = check_scope(x["q"])
    sc = res["scope"]
    top = search(x["q"], k=1, scheme="time")            # 범위 제한 없는 1위 점수 (기준값 후보용)
    rows.append({"id": x["id"], "type": x["type"], "scope": sc or "none", "expect": EXPECT.get(x["id"], "none"),
                 "error": res["error"] or "", "top1_score": round(top[0]["score"], 4) if top else None, "top1_id": top[0]["id"] if top else "", "q": x["q"]})
    print(x["id"], sc or "none")

df = pd.DataFrame(rows)
u, ok = df[df.type == "unanswerable"], df[df.type != "unanswerable"]
n_err = int((df.error != "").sum())
summary = {"tag": a.tag, "errors": n_err,
           "refuse_recall_U": round((u.scope != "none").mean(), 3), "type_ok_U": round((u.scope == u.expect).mean(), 3),
           "false_refuse_n": int((~ok.scope.isin(["none", *SOFT])).sum()),
           "false_refuse_ids": ok[~ok.scope.isin(["none", *SOFT])].id.tolist(),
           "soft_notice_ids": ok[ok.scope.isin(SOFT)].id.tolist(),
           "top1_U": u.top1_score.describe()[["min", "50%", "max"]].round(4).to_dict(),
           "top1_other": ok.top1_score.describe()[["min", "50%", "max"]].round(4).to_dict()}
out = ROOT / "eval/results"
out.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(out / f"{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
df.to_csv(out / f"{a.tag}_detail.csv", index=False, encoding="utf-8-sig")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if n_err:
    print(f"\n⚠️ 범위 판정 오류 {n_err}건 — 수치를 쓰지 말고 오류부터 확인: {df[df.error != ''].error.iloc[0]}")
print(df[df.type == "unanswerable"][["id", "scope", "expect", "top1_score"]].to_string(index=False))
print("\n[유형별 1위 점수 최솟값]\n", df.groupby("type").top1_score.min().round(4).to_string())