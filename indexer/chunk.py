"""4종 문서를 청크로 나눠 docs.db의 chunks 테이블에 저장 (기획서 5-3, 4-3)
- 댓글: 사례·영상 단위, 무관 제외, 15개씩. scheme=reason(방향·이유별로 묶음) / time(시간순). 묶음 안은 둘 다 시간순
  이유 이름은 글에 넣지 않고 메타데이터에만 (묶는 방식만 다르게, 변수 하나)
- 사례 요약: 사례당 1청크 / 뉴스: 약 500자, 문장 1개 겹침, 고지문 제거 / 정부: 쪽 단위(짧은 쪽은 다음 쪽과 합침) + 장 이름
- 비댓글 청크는 scheme=common. embed.py는 common + 한 안을 골라 적재
실행 (프로젝트 루트에서): python -m indexer.chunk --tag exp1b_casedesc
"""

import argparse, json, re, sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "store" / "docs.db"
PER_CHUNK, NEWS_LEN, GOV_MIN, COVER = 15, 500, 300, 50
MERGE = {"R0": "G08", "R8": "G08", "R6": "G67", "R7": "G67"}
AUTH = {"youtube_comment": "소비자 반응", "case_summary": "소비자 반응", "news_provided": "언론 보도", "gov": "정부 기준"}
KIND = {"youtube_comment": "유튜브 댓글", "case_summary": "사례 요약", "news_provided": "제공 뉴스", "gov": "정부 문서"}
# 장 표지 쪽 → 장 이름 (gov_pages 50자 미만 쪽 확인 결과, 11/5)
SECTIONS = {"gov_01": [(1, "앞부분"), (7, "제1장 생성형 AI 기술과 저작권"), (15, "제2장 AI 사업자에 대한 안내사항"),
                       (23, "제3장 저작권자에 대한 안내사항"), (29, "제4장 AI 이용자에 대한 안내사항"),
                       (39, "제5장 AI 산출물과 저작권 등록"), (47, "제6장 AI-저작권 제도개선 워킹그룹 운영 경과 및 향후 추진 계획"),
                       (53, "Q&A"), (71, "부록")],
            "gov_02": [(1, "앞부분"), (6, "Ⅰ 생성형 인공지능과 저작권"), (12, "Ⅱ 생성형 인공지능 학습의 저작권 침해 가능성"),
                       (24, "Ⅲ 공정이용 해설"), (56, "Ⅳ 생성형 인공지능 저작권 분쟁 가능성에 대한 대응 방안"), (62, "부록")]}
NEWS_NOISE = re.compile(r"무단\s*전재|재배포\s*금지|인용\s*보도\s*시|저작권자\s*[©ⓒ(]|저작권은\s*\S+에\s*있|AI\s*생성\s*이미지")
NAVER_UI = re.compile(r"기사를 더 보고 싶다면\?|기사 섹션 분류 안내|이 기사를 추천합니다")   # 네이버 화면 문구

def chunk(rows, doc_type, doc_id, title, case_name, text, scheme="common", **meta):
    prefix = f"[{KIND[doc_type]}] {title}" + (f" ({case_name})" if case_name else "")
    body = text if text.startswith("[사례 요약]") else f"{prefix}\n{text}"   # 사례 요약은 첫 줄이 이미 머리말
    meta = {k: v for k, v in meta.items() if v is not None and not (isinstance(v, float) and pd.isna(v))}
    n = sum(1 for r in rows if r["doc_id"] == doc_id and r["scheme"] == scheme)
    rows.append({"chunk_id": f"{doc_id}#{scheme}#{n}", "scheme": scheme, "doc_id": doc_id, "doc_type": doc_type, "text": body,
                 "meta": json.dumps({"doc_id": doc_id, "title": title, "doc_type": doc_type, "authority": AUTH[doc_type],
                                     "chunk_index": n, **meta}, ensure_ascii=False)})


def comments(rows, con, tag, cases, summ):
    vids = pd.read_csv(ROOT / "collector" / "case_videos.csv", encoding="utf-8-sig").query("use == 'Y'").copy()
    vids["video_id"] = vids.video_url.str.extract(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})")[0]
    g = pd.read_sql("""SELECT y.comment_id, y.case_id, y.video_id, y.is_reply, y.text, y.published_at, y.collected_at,
                              l.direction, l.reason FROM youtube_comments y JOIN comment_label l USING(comment_id)
                       LEFT JOIN comment_filter f USING(comment_id)
                       WHERE l.run_tag = ? AND (f.exclude_reason IS NULL OR f.exclude_reason = '') AND l.direction != '무관'""",
                    con, params=(tag,))
    g["code"] = g.reason.map(lambda r: MERGE.get(r, r))
    g["line"] = g.apply(lambda r: ("- (답글) " if r.is_reply else "- ") + " ".join(r.text.split()), axis=1)
    for v in vids.itertuples():
        c, sub = cases.loc[v.case_id], g[g.video_id == v.video_id].sort_values("published_at")
        base = dict(source=v.channel, url=v.video_url, published_at=v.published_at, collected_at=sub.collected_at.max(),
                    case_id=v.case_id, industry=c.industry, ai_part=c.ai_part, disclosed=c.disclosed, period=c.period,
                    video_kind=v.video_kind, qualified=int(summ.loc[v.case_id]))
        for scheme, groups in [("time", [((None, None), sub)]), ("reason", list(sub.groupby(["direction", "code"])))]:
            for (d, code), grp in groups:
                for i in range(0, len(grp), PER_CHUNK):
                    part = grp.iloc[i:i + PER_CHUNK]
                    chunk(rows, "youtube_comment", f"yt_{v.video_id}", v.title, c.case_name, "\n".join(part.line), scheme,
                          direction=d, reason=code, n_comments=len(part), **base)
    print(f"댓글 {len(g)}개 사용 (무관 제외)")


def summaries(rows, con, cases):
    s = pd.read_sql("SELECT case_id, run_tag, qualified, text FROM case_summary", con)
    col_at = pd.read_sql("SELECT case_id, MAX(collected_at) AS c FROM youtube_comments GROUP BY case_id", con).set_index("case_id").c
    vids = pd.read_csv(ROOT / "collector" / "case_videos.csv", encoding="utf-8-sig").query("use == 'Y'")
    for r in s.itertuples():
        c, v = cases.loc[r.case_id], vids[vids.case_id == r.case_id].iloc[0]
        chunk(rows, "case_summary", f"summary_{r.case_id}", c.case_name, c.case_name, r.text, source="댓글 분류 결과(코드 집계)", collected_at=col_at.get(r.case_id), run_tag=r.run_tag,
              url=v.video_url, published_at=v.published_at, case_id=r.case_id, industry=c.industry, ai_part=c.ai_part,
              disclosed=c.disclosed, period=c.period, qualified=int(r.qualified))


def news(rows, con):
    removed = 0
    for n in pd.read_sql("SELECT * FROM news_provided", con).itertuples():
        body = NAVER_UI.sub(" ", n.text)   # 화면 문구만 잘라냄 (문장째 지우지 않음)
        ui_removed = len(NAVER_UI.findall(n.text)); removed += ui_removed
        sents = [s for s in re.split(r"(?<=[.다])\s+", " ".join(body.split())) if s]
        keep = [s for s in sents if not NEWS_NOISE.search(s)]
        removed += len(sents) - len(keep)
        buf = []
        for s in keep:
            if buf and len(" ".join(buf + [s])) > NEWS_LEN:
                chunk(rows, "news_provided", n.doc_id, n.title, None, " ".join(buf), source=n.source, url=n.url,
                      published_at=n.published_at, collected_at=n.collected_at)
                buf = buf[-1:]   # 문장 1개 겹침
            buf.append(s)
        if buf:
            chunk(rows, "news_provided", n.doc_id, n.title, None, " ".join(buf), source=n.source, url=n.url,
                  published_at=n.published_at, collected_at=n.collected_at)
    print(f"뉴스 고지문·사진 설명으로 제거한 문장 {removed}개")


def gov(rows, con):
    docs = pd.read_sql("SELECT * FROM gov_docs", con).set_index("doc_id")
    pages = pd.read_sql("SELECT doc_id, page, text, n_chars FROM gov_pages WHERE n_chars >= ?", con, params=(COVER,))
    for doc_id, g in pages.groupby("doc_id"):
        d, sec = docs.loc[doc_id], SECTIONS[doc_id]
        name = lambda p: [s for start, s in sec if start <= p][-1]
        buf = []
        for r in g.sort_values("page").itertuples():
            if buf and name(buf[0].page) != name(r.page):   # 장이 바뀌면 합치지 않음
                flush(rows, doc_id, d, name(buf[0].page), buf); buf = []
            buf.append(r)
            if sum(b.n_chars for b in buf) >= GOV_MIN:
                flush(rows, doc_id, d, name(buf[0].page), buf); buf = []
        if buf:
            flush(rows, doc_id, d, name(buf[0].page), buf)


def flush(rows, doc_id, d, section, buf):
    chunk(rows, "gov", doc_id, d.title, None, f"<{section}>\n" + "\n".join(b.text for b in buf), source=d.source, url=d.url,
          published_at=d.published_at, collected_at=d.collected_at, section=section,
          page_start=int(buf[0].page), page_end=int(buf[-1].page))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    cases = pd.read_csv(ROOT / "collector" / "cases.csv", encoding="utf-8-sig").set_index("case_id")
    rows = []
    with sqlite3.connect(DB) as con:
        summ = pd.read_sql("SELECT case_id, qualified FROM case_summary", con).set_index("case_id").qualified
        comments(rows, con, a.tag, cases, summ); summaries(rows, con, cases); news(rows, con); gov(rows, con)
        df = pd.DataFrame(rows)
        df.to_sql("chunks", con, if_exists="replace", index=False)
    df["len"] = df.text.str.len()
    print(df.groupby(["doc_type", "scheme"]).len.agg(["count", "mean", "max"]).round(0).to_string())
    for t in df.doc_type.unique():
        print(f"\n[예시 {t}]\n" + df[df.doc_type == t].text.iloc[0][:300])


if __name__ == "__main__":
    main()