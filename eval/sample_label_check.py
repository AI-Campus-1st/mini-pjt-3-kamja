"""사람 검수용 댓글 100개를 무작위로 뽑아 eval/grading/label_check.csv로 저장 (seed 고정)

- store/docs.db를 읽기만 한다 (쓰기 없음)
- 검수 전 임시 거르기: 채널 주인 제외, 공백 뺀 10자 이상, 한글 비율 0.5 이상
  (indexer의 정식 정제와 같은 기준. 이벤트성 필터는 아직 없으므로 검수 메모로 표시)
- 사례별로 고르게 (사례 수로 100을 나눔), 사례 안에서는 답글 비율만큼 답글 포함
- 답글은 원래 댓글 본문을 parent_text 칸에 함께 넣는다 (코드북 v1.1 4-6)
- 실행 (프로젝트 루트에서): python -m eval.sample_label_check
"""

import re, sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB, OUT = ROOT / "store" / "docs.db", ROOT / "eval" / "grading" / "label_check.csv"
TOTAL, SEED = 100, 42


def ko_ratio(t):
    letters = re.findall(r"[A-Za-z가-힣\u3040-\u30ff\u4e00-\u9fff]", t)
    return sum("가" <= c <= "힣" for c in letters) / len(letters) if letters else 0


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} 가 이미 있음. 검수 결과를 덮어쓰지 않도록 멈춥니다.")
    with sqlite3.connect(DB) as con:
        df = pd.read_sql("SELECT comment_id, case_id, parent_id, is_reply, is_owner, text FROM youtube_comments", con)
    parent_text = df.set_index("comment_id")["text"]

    ok = df[(df.is_owner == 0)
            & (df.text.str.replace(r"\s", "", regex=True).str.len() >= 10)
            & (df.text.map(ko_ratio) >= 0.5)].copy()

    cases = sorted(ok.case_id.unique())

    per_case = [TOTAL // len(cases) + (i < TOTAL % len(cases)) for i in range(len(cases))]
    picks = []
    for case, n in zip(cases, per_case):
        g = ok[ok.case_id == case]
        n_reply = round(n * g.is_reply.mean())
        picks += [g[g.is_reply == 1].sample(min(n_reply, (g.is_reply == 1).sum()), random_state=SEED),
                  g[g.is_reply == 0].sample(n - min(n_reply, (g.is_reply == 1).sum()), random_state=SEED)]
    s = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)  # 사례 순서 섞기

    s["parent_text"] = s.parent_id.map(parent_text).fillna("")
    s.insert(0, "sample_id", [f"L{i+1:03d}" for i in range(len(s))])
    for col in ["human_direction", "human_reason", "memo"]:
        s[col] = ""
    s = s[["sample_id", "comment_id", "case_id", "is_reply", "parent_text", "text",
           "human_direction", "human_reason", "memo"]]
    s.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{len(s)}개 저장 → {OUT}")
    print(s.groupby("case_id").is_reply.agg(["size", "sum"]).rename(columns={"size": "전체", "sum": "답글"}))


if __name__ == "__main__":
    main()
