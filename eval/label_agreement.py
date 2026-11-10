"""사람 검수(eval/grading/label_check.csv)와 GPT 분류(comment_label)의 일치율을 전체·최상위·답글로 계산 (기획서 2-4)

- store/docs.db는 읽기만 한다. 정제에서 빠진 검수 표본은 계산에서 뺀다 (96개 기준, 2026-11-04 결정)
- 결과: eval/results/<tag>_<검수파일명>.json, 어긋난 댓글은 같은 이름_miss.csv
- 실행 (프로젝트 루트에서): python -m eval.label_agreement --tag base
  새 40개로 잴 때: python -m eval.label_agreement --tag <태그> --lab eval/grading/label_check2.csv
"""

import argparse, json, sqlite3
from pathlib import Path
import pandas as pd

# 기획서 11항 "이유 코드 합쳐 단순화". 검수 96개(base~exp1b)의 혼동만 보고 정함 (2026-11-04, 새 40개 결과 확인 전 기준)
# AI와 무관한 반응 R0+R8, AI 호평 R6+R7. 예상 등급에 쓰는 부정 이유 R1~R5는 그대로
MERGE = {"R0": "G08", "R8": "G08", "R6": "G67", "R7": "G67"}

ROOT = Path(__file__).resolve().parents[1]
DB, OUT = ROOT / "store" / "docs.db", ROOT / "eval" / "results"


def rates(g):
    if g.empty:
        return {"n": 0, "direction": None, "reason": None, "reason_merged": None, "negative": None}
    pct = lambda x: round(x.mean() * 100, 1)
    mg = lambda r: r.map(lambda c: MERGE.get(c, c))
    return {"n": len(g), "direction": pct(g.human_direction == g.direction), "reason": pct(g.human_reason == g.reason),
            "reason_merged": pct(mg(g.human_reason) == mg(g.reason)),                       # 묶은 이유 코드
            "negative": pct((g.human_direction == "부정") == (g.direction == "부정"))}        # 부정인지 아닌지만


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--lab", default="eval/grading/label_check.csv")
    a = ap.parse_args(); name = f"{a.tag}_{Path(a.lab).stem}"   # 예: base_label_check, exp1b_label_check2
    lab = pd.read_csv(ROOT / a.lab, encoding="utf-8-sig", dtype={"human_direction": str, "human_reason": str})
    if lab.human_direction.isna().any() or lab.human_reason.isna().any():   # 검수 전에 돌리면 GPT 결과를 보기 전에 멈춤
        raise SystemExit(f"{a.lab}에 사람 검수가 비어 있는 칸이 있음. 검수를 끝낸 뒤 실행하세요.")
    lab["human_direction"] = lab.human_direction.str.strip()
    lab["human_reason"] = lab.human_reason.str.strip().str.upper()
    with sqlite3.connect(DB) as con:
        f = pd.read_sql("SELECT comment_id, exclude_reason FROM comment_filter", con)
        g = pd.read_sql("SELECT comment_id, direction, reason, valid FROM comment_label WHERE run_tag=?", con, params=(a.tag,))
    m = lab.merge(f, on="comment_id").query("exclude_reason == ''").merge(g, on="comment_id", how="left")
    if m.direction.isna().any():
        raise SystemExit(f"GPT 분류가 없는 검수 표본 {m.direction.isna().sum()}개. label.py를 먼저 돌리세요.")

    res = {"tag": a.tag, "lab": a.lab, "all": rates(m), "top": rates(m[m.is_reply == 0]), "reply": rates(m[m.is_reply == 1]),
           "invalid_pair": int((m.valid == 0).sum())}
    pairs = m[m.human_reason != m.reason].groupby(["human_reason", "reason"]).size().sort_values(ascending=False)
    res["reason_confusion_top5"] = {f"사람 {h} → GPT {r}": int(n) for (h, r), n in pairs.head(5).items()}

    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    miss = m[(m.human_direction != m.direction) | (m.human_reason != m.reason)]
    miss[["sample_id", "case_id", "is_reply", "parent_text", "text", "human_direction", "human_reason",
          "direction", "reason", "memo"]].to_csv(OUT / f"{name}_miss.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"어긋난 댓글 {len(miss)}개 → {OUT / f'{name}_miss.csv'}")


if __name__ == "__main__":
    main()