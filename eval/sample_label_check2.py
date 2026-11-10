"""코드북 v2의 최종 일치율용 새 검수 40개를 뽑아 eval/grading/label_check2.csv로 저장 (seed 43)

- 대상: 정식 정제를 통과한 댓글(comment_filter.exclude_reason == ''), 기존 label_check.csv 100개는 제외
- 사례별로 고르게 (이름순 앞 7개 사례 4개, 나머지 3개), 사례 안에서는 정제 후 답글 비율만큼 답글
- store/docs.db는 읽기만 한다. 사람 검수는 전체 GPT 분류를 돌리기 전에 끝낸다
- 실행 (프로젝트 루트에서): python -m eval.sample_label_check2
"""

import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB, OLD, OUT = ROOT / "store" / "docs.db", ROOT / "eval" / "grading" / "label_check.csv", ROOT / "eval" / "grading" / "label_check2.csv"
TOTAL, SEED = 40, 43


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} 가 이미 있음. 검수 결과를 덮어쓰지 않도록 멈춥니다.")
    with sqlite3.connect(DB) as con:
        df = pd.read_sql("""SELECT y.comment_id, y.case_id, y.is_reply, y.text, p.text AS parent_text
                            FROM youtube_comments y JOIN comment_filter f USING(comment_id)
                            LEFT JOIN youtube_comments p ON p.comment_id = y.parent_id
                            WHERE f.exclude_reason = ''""", con)
    old = pd.read_csv(OLD, encoding="utf-8-sig").comment_id
    ok = df[~df.comment_id.isin(old)]

    cases = sorted(ok.case_id.unique())
    per_case = [TOTAL // len(cases) + (i < TOTAL % len(cases)) for i in range(len(cases))]
    picks = []
    for case, n in zip(cases, per_case):
        g = ok[ok.case_id == case]
        n_reply = min(round(n * g.is_reply.mean()), int(g.is_reply.sum()))
        picks += [g[g.is_reply == 1].sample(n_reply, random_state=SEED),
                  g[g.is_reply == 0].sample(n - n_reply, random_state=SEED)]
    s = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)   # 사례 순서 섞기

    s["parent_text"] = s.parent_text.fillna("")
    s.insert(0, "sample_id", [f"N{i+1:03d}" for i in range(len(s))])   # 기존 L001~과 구분
    for col in ["human_direction", "human_reason", "memo"]:
        s[col] = ""
    s = s[["sample_id", "comment_id", "case_id", "is_reply", "parent_text", "text",
           "human_direction", "human_reason", "memo"]]
    s.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{len(s)}개 저장 → {OUT} (기존 100개와 겹침 {s.comment_id.isin(old).sum()}개)")
    print(s.groupby("case_id").is_reply.agg(["size", "sum"]).rename(columns={"size": "전체", "sum": "답글"}))


if __name__ == "__main__":
    main()
