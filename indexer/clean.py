"""댓글을 정제해 제외 이유를 docs.db의 comment_filter 테이블에 저장하고, 영상별 정제 전후 건수를 기록 (기획서 5-2)

- 원본 youtube_comments는 바꾸지 않는다. 결과는 comment_filter(comment_id, exclude_reason)에 따로 쓴다
  exclude_reason이 빈 칸이면 남긴 댓글, 아니면 처음 걸린 기준 하나
- 기준 순서: owner → short(10자 미만) → foreign(한글 비율 0.5 미만) → event(이벤트·홍보성 의심) → dup(같은 사례 안 같은 문구)
- 글자 수·한글 비율은 @사용자와 보이지 않는 공백 문자를 지운 뒤 잰다
- 실행 (프로젝트 루트에서): python -m indexer.clean
"""

import re, sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB, STATS = ROOT / "store" / "docs.db", ROOT / "report" / "evidence" / "clean_stats.csv"

INVISIBLE = r"[\u200b-\u200f\u2060\ufeff\u00ad]"   # 폭 없는 공백, 방향 표시, BOM, 소프트 하이픈
TIMESTAMP = r"^\s*\d{1,2}:\d{2}"                   # 0:15 처럼 시간으로 시작
CHEER_END = r"(응원|화이팅|파이팅|홧팅|번창)(합니다|해요|할게요|하겠습니다)?[^가-힣]*$"


def measure_text(t):
    """길이·한글 비율을 잴 때 쓰는 글: @사용자, 보이지 않는 문자 제거"""
    return re.sub(INVISIBLE, "", t.replace("@사용자", ""))


def ko_ratio(t):
    letters = re.findall(r"[A-Za-z가-힣\u3040-\u30ff\u4e00-\u9fff]", t)
    return sum("가" <= c <= "힣" for c in letters) / len(letters) if letters else 0


def is_event(t):
    """타임스탬프로 시작 + 응원으로 끝남 (기획서 5-2 패턴). 단어만으로는 거르지 않음: 11/4 점검에서 '이벤트' 단어 5건이 모두 일반 댓글이었음"""
    return bool(re.search(TIMESTAMP, t) and re.search(CHEER_END, t))


def dup_key(t):
    """문구 중복 판정용: 공백·문장부호·ㅋㅎ 반복을 지운 글"""
    return re.sub(r"[^가-힣A-Za-z0-9]|[ㅋㅎㅠㅜ]+", "", t)


def main():
    with sqlite3.connect(DB) as con:
        df = pd.read_sql("SELECT * FROM youtube_comments", con)
    keys = ["case_id", "video_id"] if "video_id" in df.columns else ["case_id"]   # 영상 열이 없으면 사례 단위로 집계

    m = df.text.fillna("").map(measure_text)
    df["exclude_reason"] = ""
    rules = [("owner", df.is_owner == 1),
             ("short", m.str.replace(r"\s", "", regex=True).str.len() < 10),
             ("foreign", m.map(ko_ratio) < 0.5),
             ("event", m.map(is_event))]
    for name, mask in rules:   # 앞 기준에 걸리지 않은 것만 다음 기준을 적용
        df.loc[mask & (df.exclude_reason == ""), "exclude_reason"] = name

    # 같은 사례 안 같은 문구: 최상위 댓글을 먼저, 그다음 좋아요가 많은 1개만 남김 (사례 자격은 최상위로 세므로)
    df["_key"] = m.map(dup_key)
    alive = df[df.exclude_reason == ""].sort_values(["is_reply", "like_count"], ascending=[True, False])
    dups = alive[alive.duplicated(["case_id", "_key"], keep="first")].index
    df.loc[dups, "exclude_reason"] = "dup"

    with sqlite3.connect(DB) as con:
        df[["comment_id", "exclude_reason"]].to_sql("comment_filter", con, if_exists="replace", index=False)

    # 영상별·최상위/답글별 정제 전후 건수
    df["kind"] = df.is_reply.map({0: "최상위", 1: "답글"})
    stats = pd.crosstab([df[k] for k in keys] + [df.kind], df.exclude_reason.replace("", "kept"))
    stats.insert(0, "before", stats.sum(axis=1))
    stats["exclude_pct"] = (1 - stats.get("kept", 0) / stats.before).mul(100).round(1)
    STATS.parent.mkdir(exist_ok=True)
    stats.to_csv(STATS, encoding="utf-8-sig")

    print(f"정제 전 {len(df):,}개 → 남은 {(df.exclude_reason == '').sum():,}개")
    print(df.exclude_reason.replace("", "kept").value_counts().to_string())
    top = df[(df.is_reply == 0) & (df.exclude_reason == "")].groupby("case_id").size()
    print("\n[사례별 남은 최상위 댓글 (무관 분류 전)]\n" + top.to_string())
    print(f"\n영상별 표 → {STATS}")


if __name__ == "__main__":
    main()