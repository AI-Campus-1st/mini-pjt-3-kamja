"""유튜브 댓글 수집기 (collector 계층)

- collector/case_videos.csv 에서 use=Y 영상만 수집한다
- 최상위 댓글은 인기순(relevance)으로 상한까지, 그 댓글의 답글은 전부 받는다
- 작성자 이름·채널 ID·프로필은 저장하지 않는다. 채널 주인 여부(is_owner)만 비교해 남긴다
- 답글 속 @닉네임은 저장 전에 @사용자로 바꾼다 (개인정보 마스킹)
- store/docs.db 의 youtube_comments 테이블에 comment_id 기준으로 저장 (재수집 시 중복 없이 갱신)

실행 (프로젝트 루트에서):
    python -m collector.collect_youtube              # 전체
    python -m collector.collect_youtube --only case_01   # 한 사례만 테스트
"""


import argparse, os, re, sqlite3, time, unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
VIDEOS_CSV = ROOT / "collector" / "case_videos.csv"
DB_PATH = ROOT / "store" / "docs.db"
API = "https://www.googleapis.com/youtube/v3"

DEFAULT_CAP = 500                 # 최상위 댓글 상한
CAP_OVERRIDE = {"case_10": 2000}  # 시프트업(브랜드명) : 한국어 필터 후 30개 이상 확보용
ORDER = "relevance"               # API 기본값은 time 이므로 명시
MENTION = re.compile(r"@[\w가-힣.\-]+")


class QuotaExceeded(Exception):
    pass


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def video_id_from(url):
    m = re.search(r"(?:v=|/shorts/|youtu\.be/)([\w-]{11})", url)
    if not m:
        raise ValueError(f"영상 ID를 찾을 수 없음: {url}")
    return m.group(1)


def get(endpoint, key, **params):
    """API 요청 1회. 할당량 초과면 멈추고, 그 외 오류는 사유를 돌려준다"""
    r = requests.get(f"{API}/{endpoint}", params={**params, "key": key}, timeout=30)
    if r.ok:
        return r.json(), None
    reason = r.json().get("error", {}).get("errors", [{}])[0].get("reason", str(r.status_code))
    if reason in ("quotaExceeded", "dailyLimitExceeded"):
        raise QuotaExceeded(reason)
    return None, reason


def to_row(c, case_id, video_id, owner_ch, collected_at, parent_id=None):
    """API 댓글 1개 → 저장할 행 (작성자 정보는 비교에만 쓰고 버림)"""
    s = c["snippet"]
    author_ch = s.get("authorChannelId", {}).get("value")
    text = nfc(s.get("textOriginal") or s.get("textDisplay"))
    if parent_id:
        text = MENTION.sub("@사용자", text)
    return dict(
        comment_id=c["id"], case_id=case_id, video_id=video_id,
        parent_id=parent_id, is_reply=int(bool(parent_id)),
        is_owner=int(author_ch is not None and author_ch == owner_ch),
        text=text, like_count=s.get("likeCount", 0),
        published_at=s.get("publishedAt"), sort_order=ORDER, collected_at=collected_at,
    )


def fetch_replies(key, parent_id, **ctx):
    rows, token = [], None
    while True:
        data, err = get("comments", key, part="snippet", parentId=parent_id,
                        maxResults=100, textFormat="plainText", pageToken=token)
        if err:
            print(f"    답글 실패 {parent_id}: {err}")
            return rows
        rows += [to_row(c, parent_id=parent_id, **ctx) for c in data["items"]]
        token = data.get("nextPageToken")
        if not token:
            return rows


def collect_video(key, case_id, url, cap):
    video_id = video_id_from(url)
    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    info, err = get("videos", key, part="snippet", id=video_id)
    if err or not info["items"]:
        return [], f"영상 조회 실패: {err or '삭제/비공개'}"
    ctx = dict(case_id=case_id, video_id=video_id, collected_at=collected_at,
               owner_ch=info["items"][0]["snippet"]["channelId"])

    rows, top, token = [], 0, None
    while top < cap:
        data, err = get("commentThreads", key, part="snippet", videoId=video_id, order=ORDER,
                        maxResults=min(100, cap - top), textFormat="plainText", pageToken=token)
        if err:
            return rows, f"댓글 목록 실패: {err}"
        for t in data["items"]:
            rows.append(to_row(t["snippet"]["topLevelComment"], **ctx))
            top += 1
            if t["snippet"].get("totalReplyCount", 0) > 0:
                rows += fetch_replies(key, t["id"], **ctx)
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)
    return rows, None


SCHEMA = """CREATE TABLE IF NOT EXISTS youtube_comments (
    comment_id TEXT PRIMARY KEY, case_id TEXT, video_id TEXT, parent_id TEXT,
    is_reply INTEGER, is_owner INTEGER, text TEXT, like_count INTEGER,
    published_at TEXT, sort_order TEXT, collected_at TEXT)"""
UPSERT = """INSERT INTO youtube_comments VALUES (:comment_id, :case_id, :video_id, :parent_id,
    :is_reply, :is_owner, :text, :like_count, :published_at, :sort_order, :collected_at)
    ON CONFLICT(comment_id) DO UPDATE SET text=excluded.text, like_count=excluded.like_count,
    collected_at=excluded.collected_at"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="이 case_id만 수집 (테스트용)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ["YOUTUBE_API_KEY"]
    videos = pd.read_csv(VIDEOS_CSV, encoding="utf-8-sig", dtype=str)
    videos = videos[videos["use"].str.strip() == "Y"]
    if args.only:
        videos = videos[videos["case_id"] == args.only]

    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(SCHEMA)
    try:
        for v in videos.itertuples():
            cap = CAP_OVERRIDE.get(v.case_id, DEFAULT_CAP)
            rows, err = collect_video(key, v.case_id, v.video_url, cap)
            with con:
                con.executemany(UPSERT, rows)
            reply = sum(r["is_reply"] for r in rows)
            owner = sum(r["is_owner"] for r in rows)
            print(f"{v.case_id} {v.video_url[-11:]}: 최상위 {len(rows)-reply}, 답글 {reply}, "
                  f"채널주인 {owner}" + (f"  ⚠ {err}" if err else ""))
    except QuotaExceeded as e:
        print(f"할당량 초과로 중단: {e}")
    finally:
        total = con.execute("SELECT COUNT(*) FROM youtube_comments").fetchone()[0]
        print(f"DB 전체 댓글 수: {total}")
        con.close()


if __name__ == "__main__":
    main()