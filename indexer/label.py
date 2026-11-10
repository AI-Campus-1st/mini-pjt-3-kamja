"""정제 후 남은 댓글을 GPT로 분류해 docs.db의 comment_label 테이블에 저장 (기획서 2-3, 코드북 v1.1)

- 대상: comment_filter.exclude_reason이 빈 칸인 댓글. --only를 주면 그 CSV의 comment_id만 (검수 96개 먼저)
- 입력: 코드북 1~5항(고정, 앞부분 캐시용) + 사례 카드 + 댓글. 답글은 "원래 댓글: … / 답글: …" 형태
- 출력 검사: 방향·이유 짝 규칙. 어긋나면 1번 다시 묻고, 그래도 어긋나면 valid=0으로 기록
- 같은 run_tag로 이미 분류한 댓글은 건너뜀 (중간에 끊겨도 이어서 실행)
- 실행 (프로젝트 루트에서):
    python -m indexer.label --tag base --only eval/grading/label_check.csv   # 검수 96개
    python -m indexer.label --tag base                               # 전체
    --casedesc: cases.csv의 ad_desc(사람이 영상을 보고 쓴 광고 내용 한 줄)를 함께 넣음 (exp1b_casedesc)
"""

import argparse, json, sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "store" / "docs.db"
MODEL, WORKERS = "gpt-4.1-mini", 5
PRICE = {"in": 0.40, "cached": 0.10, "out": 1.60}   # $/100만 토큰, OpenAI 공식 가격표 2026-11-04 확인

DIRS = {"긍정", "부정", "중립", "무관"}
CODEBOOK = (ROOT / "indexer" / "codebook.md").read_text(encoding="utf-8").split("## 6.")[0]
CODEBOOK_VER = CODEBOOK.splitlines()[0].split("코드북")[-1].strip()   # 첫 줄 "# 댓글 분류 코드북 v2"에서 버전 읽기
SYSTEM = ("너는 유튜브 광고 영상 댓글을 아래 코드북에 따라 분류한다. 코드북의 모든 규칙(짝 규칙, 경계 규칙, 답글 규칙)을 지킨다.\n"
          '반드시 JSON 한 줄로만 답한다: {"direction": "...", "reason": "R0~R8"}\n\n' + CODEBOOK)


def pair_ok(d, r):
    """코드북 3항 짝 규칙 검사"""
    if d not in DIRS or r not in {f"R{i}" for i in range(9)}:
        return False
    if d == "무관":
        return r == "R0"
    return not ((r in {"R1", "R2", "R3", "R4", "R5"} and d != "부정") or (r in {"R6", "R7"} and d != "긍정"))


def user_msg(row):
    head = f"[사례] {row.case_name} / 업종: {row.industry} / AI를 쓴 부분: {row.ai_part}\n"
    if getattr(row, "ad_desc", ""):
        head += f"[광고 내용] {row.ad_desc}\n"
    if row.is_reply:
        return head + f"원래 댓글: {row.parent_text}\n답글: {row.text}\n(답글만 분류)"
    return head + f"댓글: {row.text}"


def classify(client, row):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(row)}]
    usage = {"in": 0, "cached": 0, "out": 0}
    for attempt in range(2):
        r = client.chat.completions.create(model=MODEL, temperature=0, messages=msgs,
                                           response_format={"type": "json_object"})
        u = r.usage
        cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        usage["in"] += u.prompt_tokens - cached; usage["cached"] += cached; usage["out"] += u.completion_tokens
        out = r.choices[0].message.content
        try:
            j = json.loads(out); d, rs = j.get("direction", ""), str(j.get("reason", "")).upper()
        except json.JSONDecodeError:
            d, rs = "", ""
        if pair_ok(d, rs):
            return row.comment_id, d, rs, 1, attempt, usage
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": "짝 규칙(R1~R5는 부정, R6·R7은 긍정, 무관은 R0)에 맞지 않는다. 다시 JSON으로만 답하라."}]
    return row.comment_id, d, rs, 0, attempt, usage


def load_targets(con, tag, only, casedesc):
    df = pd.read_sql("""SELECT y.comment_id, y.case_id, y.is_reply, y.text, p.text AS parent_text
                        FROM youtube_comments y JOIN comment_filter f USING(comment_id)
                        LEFT JOIN youtube_comments p ON p.comment_id = y.parent_id
                        WHERE f.exclude_reason = ''""", con)
    cases = pd.read_csv(ROOT / "collector" / "cases.csv", encoding="utf-8-sig")
    cols = ["case_id", "case_name", "industry", "ai_part"] + (["ad_desc"] if casedesc else [])
    if casedesc and cases.ad_desc.isna().any():
        raise SystemExit(f"ad_desc가 빈 사례: {list(cases.case_id[cases.ad_desc.isna()])}")
    df = df.merge(cases[cols], on="case_id")
    if only:
        df = df[df.comment_id.isin(pd.read_csv(ROOT / only, encoding="utf-8-sig").comment_id)]
    con.execute("""CREATE TABLE IF NOT EXISTS comment_label (comment_id TEXT, run_tag TEXT, direction TEXT,
                   reason TEXT, valid INT, retried INT, model TEXT, codebook TEXT, labeled_at TEXT,
                   PRIMARY KEY (comment_id, run_tag))""")
    done = pd.read_sql("SELECT comment_id FROM comment_label WHERE run_tag = ?", con, params=(tag,)).comment_id
    return df[~df.comment_id.isin(done)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--only"); ap.add_argument("--casedesc", action="store_true")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env"); client = OpenAI(max_retries=6,   # 속도 제한에 걸리면 기다렸다 다시 보냄
                    default_headers={"Accept-Encoding": "gzip, deflate"})   # br·zstd 압축 응답을 받지 않음 (맥 라이브러리 버전 충돌 회피)

    with sqlite3.connect(DB) as con:
        todo = load_targets(con, a.tag, a.only, a.casedesc)
        print(f"[{a.tag}] 분류할 댓글 {len(todo):,}개 ({MODEL}, 코드북 {CODEBOOK_VER}, 광고 내용 {'포함' if a.casedesc else '없음'})")
        total = {"in": 0, "cached": 0, "out": 0}; buf, t0 = [], datetime.now()
        with ThreadPoolExecutor(WORKERS) as ex:
            for i, (cid, d, r, ok, retried, u) in enumerate(ex.map(lambda row: classify(client, row),
                                                                  todo.itertuples()), 1):
                buf.append((cid, a.tag, d, r, ok, retried, MODEL, CODEBOOK_VER, datetime.now().isoformat(timespec="seconds")))
                for k in total: total[k] += u[k]
                if len(buf) >= 50 or i == len(todo):   # 50개마다 저장 (끊겨도 이어서)
                    con.executemany("INSERT OR REPLACE INTO comment_label VALUES (?,?,?,?,?,?,?,?,?)", buf); con.commit(); buf = []
                    print(f"  {i:,}/{len(todo):,}  ({(datetime.now() - t0).seconds}초)")

        cost = sum(total[k] * PRICE[k] for k in total) / 1e6
        print(f"토큰: 입력 {total['in']:,} / 캐시 {total['cached']:,} / 출력 {total['out']:,} → 약 ${cost:.2f}")
        print(pd.read_sql("SELECT valid, COUNT(*) n, SUM(retried) retried FROM comment_label WHERE run_tag=? GROUP BY valid",
                          con, params=(a.tag,)).to_string(index=False))


if __name__ == "__main__":
    main()