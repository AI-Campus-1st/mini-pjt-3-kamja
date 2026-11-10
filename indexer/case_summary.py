"""분류 결과로 사례 요약 문서를 코드로 만들어 docs.db의 case_summary 테이블에 저장 (LLM 사용 안 함, 기획서 2-1 ⑤)

- 최상위 댓글과 답글을 나눠 집계. 자격·주된 이유는 최상위 기준 (기획서 5-2)
- 사례 자격: 무관 아닌 최상위 30개 이상. 미달 사례도 저장하되 qualified=0, "예상 근거로 쓰지 않음" 표시 (11/4 결정 A)
- 이유 코드: 부정 이유 R1~R5는 그대로, R0+R8 → G08, R6+R7 → G67 (기획서 11항 단순화, 11/4 결정)
- 주된 이유는 그 방향 최상위 댓글이 MIN_FOR_MAIN개 이상일 때만 낸다
- 실행 (프로젝트 루트에서): python -m indexer.case_summary --tag exp1b_casedesc
"""

import argparse, json, sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB, OUT = ROOT / "store" / "docs.db", ROOT / "report" / "evidence" / "case_summary.csv"
MIN_QUAL, MIN_FOR_MAIN = 30, 5
MERGE = {"R0": "G08", "R8": "G08", "R6": "G67", "R7": "G67"}
NAME = {"R1": "신뢰 하락", "R2": "오인·속은 느낌", "R3": "거부감·어색함", "R4": "비용 절감 비판",
        "R5": "창작자·저작권 우려", "G67": "AI 호평(신기함·완성도)", "G08": "AI와 무관한 반응(광고 내용·이유 없음)"}
AGREE = "분류 정확도(사람 검수 새 40개 기준): 부정 여부 85.0%, 이유(묶음) 67.5%"
LIMITS = {"case_01": "뉴스 리포트에 다른 AI 모델 사례가 함께 나와 댓글이 섞일 수 있음",
          "case_06": "광고가 아닌 방송 영상",
          "case_08": "채널이 댓글마다 답글을 다는 운영 방식이라 우호적 댓글이 많을 수 있음",
          "case_10": "해외 시청자 위주 영상의 한국어 댓글만 사용, 게임 팬 집단의 반응"}


def main_reason(rel, direction):
    """그 방향 댓글에서 가장 많은 이유. 댓글이 MIN_FOR_MAIN개 미만이면 None"""
    s = rel[rel.direction == direction].code.value_counts()
    if s.sum() < MIN_FOR_MAIN:
        return None, int(s.sum())
    m = {"code": s.index[0], "n": int(s.iloc[0]), "pct": round(s.iloc[0] / s.sum() * 100, 1)}
    if len(s) > 1:   # 다음 이유도 함께 (G08이 1위일 때 AI 관련 이유가 가려지지 않게)
        m["second"] = {"code": s.index[1], "n": int(s.iloc[1]), "pct": round(s.iloc[1] / s.sum() * 100, 1)}
        m["tie"] = bool(s.iloc[0] == s.iloc[1])   # 1·2위 개수가 같으면 동률 표시
    return m, int(s.sum())


def summarize(case, g, videos):
    top, rep = g[g.is_reply == 0], g[g.is_reply == 1]
    rel, rep_rel = top[top.direction != "무관"], rep[rep.direction != "무관"]
    pct = lambda d, x: round((x.direction == d).mean() * 100, 1) if len(x) else 0.0
    neg, n_neg = main_reason(rel, "부정")
    pos, n_pos = main_reason(rel, "긍정")
    st = {"case_id": case.case_id, "qualified": int(len(rel) >= MIN_QUAL), "top_n": len(top), "top_related": len(rel),
          "pos_pct": pct("긍정", rel), "neg_pct": pct("부정", rel), "neu_pct": pct("중립", rel),
          "neg_main": neg, "pos_main": pos, "reply_n": len(rep),
          "reply_pos_pct": pct("긍정", rep_rel), "reply_neg_pct": pct("부정", rep_rel)}

    def reason_line(label, m, n):
        if m is None:
            return f"{label} 댓글의 주된 이유: {label} 댓글이 {n}개뿐이라 정하지 않음"
        line = f"{label} 댓글의 주된 이유: {NAME[m['code']]} ({label} {n}개 중 {m['n']}개, {m['pct']}%)"
        if "second" in m:
            line += " (1·2위 동률)" if m["tie"] else ""
            line += f", 다음 이유: {NAME[m['second']['code']]} ({m['second']['n']}개, {m['second']['pct']}%)"
        return line

    def quotes(m, d):   # 주된 이유의 대표 댓글 2개 (좋아요 순, 앞 100자)
        if m is None:
            return []
        q = rel[(rel.direction == d) & (rel.code == m["code"])].nlargest(2, "like_count")
        return [f'  - "{" ".join(t.split())[:100]}"' for t in q.text]   # 줄바꿈을 한 칸으로

    lines = [f"[사례 요약] {case.case_name} ({case.brand})",
             f"특징: 업종 {case.industry} / AI를 쓴 부분 {case.ai_part} / 공개 여부 {case.disclosed} / 시기 {case.period}",
             *([f"광고 내용: {case.ad_desc}"] if isinstance(getattr(case, "ad_desc", None), str) else []),   # 영상·원기사 근거 한 줄 (11/4 결정)
             *[f"근거 영상: [{v.video_kind}] {v.title} ({v.video_url})" for v in videos.itertuples()],
             "자격: 예상 근거로 사용" if st["qualified"] else
             f"자격: 표본 부족 (무관 아닌 최상위 댓글 {len(rel)}개 < {MIN_QUAL}개). 예상 근거로 쓰지 않음, 비율은 참고용",
             f"최상위 댓글 {len(top)}개 (무관 제외 {len(rel)}개): 긍정 {st['pos_pct']}% · 부정 {st['neg_pct']}% · 중립 {st['neu_pct']}%",
             reason_line("부정", neg, n_neg), *quotes(neg, "부정"),
             reason_line("긍정", pos, n_pos), *quotes(pos, "긍정"),
             f"답글 {len(rep)}개 (따로 집계, 예상에 쓰지 않음): 긍정 {st['reply_pos_pct']}% · 부정 {st['reply_neg_pct']}%",
             AGREE,
             "주의: 유튜브 댓글은 영상을 보고 댓글을 단 일부 시청자의 반응이며, 반응이 강한 사람이 댓글을 많이 달아 실제보다 부정적으로 보일 수 있음"]
    if case.case_id in LIMITS:
        lines.append(f"한계: {LIMITS[case.case_id]}")
    return st, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    with sqlite3.connect(DB) as con:
        g = pd.read_sql("""SELECT y.case_id, y.is_reply, y.like_count, y.text, l.direction, l.reason FROM youtube_comments y
                           JOIN comment_label l USING(comment_id) WHERE l.run_tag = ?""", con, params=(a.tag,))
    g["code"] = g.reason.map(lambda r: MERGE.get(r, r))
    cases = pd.read_csv(ROOT / "collector" / "cases.csv", encoding="utf-8-sig")
    vids = pd.read_csv(ROOT / "collector" / "case_videos.csv", encoding="utf-8-sig").query("use == 'Y'")

    rows = []
    for case in cases.itertuples():
        st, text = summarize(case, g[g.case_id == case.case_id], vids[vids.case_id == case.case_id])
        rows.append({"case_id": case.case_id, "run_tag": a.tag, "qualified": st["qualified"],
                     "stats": json.dumps(st, ensure_ascii=False), "text": text})
    df = pd.DataFrame(rows)
    with sqlite3.connect(DB) as con:
        df.to_sql("case_summary", con, if_exists="replace", index=False)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"사례 요약 {len(df)}개 저장 (자격 있음 {df.qualified.sum()}개) → docs.db case_summary, {OUT}")
    for r in df.itertuples():
        st = json.loads(r.stats)
        print(f"  {r.case_id} 자격 {'O' if r.qualified else 'X'} / 부정 {st['neg_pct']}% / 부정 주된 이유 "
              f"{NAME[st['neg_main']['code']] if st['neg_main'] else '-'}")
    print("\n[예시]\n" + df.text.iloc[0])


if __name__ == "__main__":
    main()
