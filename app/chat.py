"""챗봇 화면: 평가 하네스와 같은 rag/chain.py를 부름 (prepare → stream_body). 조회만 하고 인덱스에는 쓰지 않음
화면 구성 (가이드 4-6 답변 화면·4-7 챗봇 UI 요건 대응)
- 왼쪽 사이드바(접기 가능): 새 대화, 필터(기간·문서 종류·업종 — Day 4 오후 5번에서 검색에 연결)
- 가운데: 대화(첫 화면은 입력창 바로 위에 예시 질문 블록, 누르면 바로 질문). 진행 상태 → 머리말(코드) → 본문(검사 통과 문장만 스트리밍) → 주의. 거절·빈 결과는 안내 상자
- 오른쪽(첫 답변 뒤 나타남): 고른 답변의 근거 카드 — 근거 사례 표 / 인용 댓글 / 출처(종류·제목·발행처·발행일·링크·원문·유사도)"""
import html

import pandas as pd
import streamlit as st

from rag.chain import NOT_FOUND, prepare, stream_body
from rag.retriever import _col, _model
from rag.rewrite import summary
from rag.filters import KINDS, YEARS, industries, normalize

COMPACT = ["사례", "댓글 수", "부정 %", "긍정 %"]      # 오른쪽 좁은 칸에 먼저 보이는 열 (나머지는 '더보기' 팝업)
ICON = {"부정": ":red[:material/thumb_down:]", "긍정": ":green[:material/thumb_up:]"}
EXAMPLES = ["식품 광고 영상 전체를 AI로 만들고 AI 사용을 밝히면 소비자 반응은?",
            "일부만 AI로 만든 광고와 전체를 AI로 만든 광고의 반응 차이는?",
            "AI로 만든 광고 이미지도 저작권 등록을 할 수 있나?"]

CSS = """<style>
.stApp, [data-testid="stHeader"] {background:#F6F7F9;}
[class*="st-key-card_"] {background:#FFFFFF; border:1px solid #E6E8EB; border-radius:14px; padding:1rem 1.1rem;}
[class*="st-key-card_"] [data-testid="stExpander"] details {border:none;}
.card-h {font-size:0.95rem; font-weight:700; color:#1F2329; margin:0 0 0.3rem;}
.meta {font-size:0.78rem; color:#8B9096; line-height:1.5; margin:0;}
.src-t {font-size:0.88rem; font-weight:600; color:#1F2329; line-height:1.45; margin:0.6rem 0 0.15rem;}
/* 인용 댓글: 댓글(크고 굵게) + 바로 아래 광고명(작고 옅게)을 한 문단으로 */
.qt {font-size:1.02rem; font-weight:700; color:#1F2329; line-height:1.55;}
[class*="st-key-sbtn_"] {gap:0.2rem; margin-bottom:0.3rem;}
[class*="st-key-sbtn_"] button, [class*="st-key-sbtn_"] a {min-height:0; padding:0.05rem 0.35rem;}
[class*="st-key-sbtn_"] p, [class*="st-key-more_"] p {font-size:0.78rem !important;}
/* 예시 질문 블록: 입력창 바로 위, 카드처럼 (누르면 바로 질문) */
[class*="st-key-exrow"] {gap:0.6rem; margin-bottom:0.4rem;}
[class*="st-key-ex_"] {flex:1 1 0; min-width:0;}
[class*="st-key-ex_"] button {width:100%; height:100%; justify-content:flex-start; text-align:left; background:#FFFFFF;
  border:1px solid #E6E8EB; border-radius:12px; padding:0.7rem 0.9rem; box-shadow:none;}
[class*="st-key-ex_"] button:hover {border-color:#E8877E; background:#FDF3F1;}
[class*="st-key-ex_"] button p {font-size:0.9rem; line-height:1.45; white-space:normal; color:#1F2329;}
</style>"""


@st.cache_resource(show_spinner="검색 모델 불러오는 중 (처음 한 번, 약 12초)")
def warm_up():
    _model(), _col()                  # 읽기 전용 컬렉션 열기 + 질문 임베딩 모델
    return True


def md(t):
    """마크다운 오작동 방지: ~ 는 취소선, $ 는 수식으로 바뀌므로 글자 그대로 표시"""
    return str(t).replace("~", "\\~").replace("$", "\\$")


def lines(t):
    return md(t).replace("\n", "  \n")


def card_h(text):
    st.markdown(f'<p class="card-h">{text}</p>', unsafe_allow_html=True)


# ---------- 가운데: 대화 ----------

def show_main(r, body_slot=None, skip_top=False):
    """답변 본문 (근거는 오른쪽). skip_top: 스트리밍 때 머리말을 이미 그렸으면 건너뜀"""
    if not skip_top:
        understood(r)
    if r["refusal"]:                                        # 범위 밖: 거절 문구만
        st.info(md(r["head"]), icon=":material/block:")
        return
    if not skip_top:
        if r["notice"]:                                     # legal 약한 거절: 문구 + 답변 계속
            st.warning(md(r["notice"]), icon=":material/gavel:")
        if r["head"]:
            st.markdown(lines(r["head"]))
    slot = body_slot or st.empty()
    if r.get("body") == NOT_FOUND or (not r.get("body") and not r["head"]):
        slot.info("제공된 자료에서 확인할 수 없습니다. 질문을 바꾸거나 필터를 풀어 보세요.", icon=":material/search_off:")
    elif r.get("body"):
        slot.markdown(lines(r["body"]))
    if r["foot"]:
        st.caption(md(r["foot"]))


def understood(r):
    """멀티턴: 질문을 다시 썼으면 '이렇게 이해했어요' 한 줄"""
    rw = r.get("rewrite") or {}
    if rw.get("rewritten"):
        st.caption(f"이렇게 이해했어요: {md(rw['question'])}")
    if r.get("filters"):                                     # 어떤 필터로 답했는지
        st.caption(md(r["filters"]))


def ask(q):
    """질문 하나 처리: (앞 대화로 질문 정리) → 진행 상태 → 머리말 → 스트리밍 본문 → 검사 끝난 최종 본문"""
    history = [{"q": t["question"], "a": summary(t)} for t in st.session_state.turns]   # question = 다시 쓴 질문
    with st.status("답변 준비 중", expanded=False) as status:
        p = prepare(q, progress=lambda m: status.update(label=m), history=history,
                    filters=st.session_state.get("filters"))
        status.update(label="자료 확인 끝", state="complete")
    understood(p)
    if p["refusal"]:
        show_main(p)
        return p
    if p["notice"]:
        st.warning(md(p["notice"]), icon=":material/gavel:")
    if p["head"]:
        st.markdown(lines(p["head"]))
    slot, shown, out = st.empty(), "", {}
    for piece in stream_body(p, out):                       # 검사를 통과한 문장만 흘러나옴
        shown += piece
        slot.markdown(lines(shown) + " ▌")
    r = {**p, **out}
    slot.empty()
    show_main(r, body_slot=slot, skip_top=True)
    return r


# ---------- 오른쪽: 근거 카드 ----------

@st.dialog("근거 사례 전체", width="large")
def cases_dialog(df, notes):
    st.dataframe(df, hide_index=True, width="stretch")
    for n in notes:
        st.caption(n)


@st.dialog("출처 원문", width="large")
def source_dialog(s):
    st.markdown(f"**[{s['번호']}] {md(s['제목'])}**")
    for c in s["청크"]:
        st.caption(f"유사도 {c['score']}" if c["score"] is not None else "사례 요약 (검색이 아니라 직접 조회)")
        st.text(c["text"])


def pick_quotes(quotes):
    """대표 댓글 (코드 규칙): 부정 쪽·긍정 쪽이 둘 다 있으면 각 방향 첫 사례(유사도 순) 하나씩, 한쪽뿐이면 하나"""
    return [next(g for g in quotes if g["dir"] == d) for d in ("부정", "긍정") if any(g["dir"] == d for g in quotes)]


def show_side(r, t):
    """t: 답변 번호 (버튼·카드 이름이 겹치지 않게)"""
    if r["refusal"]:
        st.caption("거절한 질문이라 근거 자료가 없습니다.")
        return
    if r["cases"]:
        with st.container(key=f"card_cases_{t}"):
            card_h("근거 사례")
            df = pd.DataFrame(r["cases"])
            df["사례"] = df["번호"] + " " + df["사례"]
            df = df.drop(columns="번호")
            st.dataframe(df, hide_index=True, width="stretch", column_order=[c for c in COMPACT if c in df])
            ex = r.get("excluded_names") or r["excluded"]
            notes = [r["accuracy"]] + (["댓글 표본이 부족해 근거에서 뺀 사례: " + ", ".join(ex)] if ex else [])
            with st.container(horizontal=True, horizontal_alignment="right", key=f"more_{t}"):
                if st.button("더보기", key=f"cases_{t}", type="tertiary"):
                    cases_dialog(df, notes)                         # 특징·주된 이유·정확도·뺀 사례까지
    quotes = [g for g in r.get("case_log", []) if g.get("quote")]
    if quotes:
        with st.container(key=f"card_quotes_{t}"):
            card_h("인용 댓글")
            n = {d: sum(g["dir"] == d for g in quotes) for d in ("부정", "긍정")}
            st.markdown(f'<p class="meta">총 {len(quotes)}개 댓글<br>부정 : {n["부정"]}개<br>긍정 : {n["긍정"]}개</p>',
                        unsafe_allow_html=True)
            for k, g in enumerate(pick_quotes(quotes)):
                short = g["quote"] if len(g["quote"]) <= 50 else g["quote"][:50] + "…"
                st.markdown(f'{ICON[g["dir"]]} <span class="qt">“{html.escape(short)}”</span><br>'   # 댓글 크고 굵게
                            f'<span class="meta">[{g["ref"]}] {html.escape(g["title"])}</span>',     # 광고명 바로 아래 작고 옅게
                            unsafe_allow_html=True)
            with st.expander(f"전체 인용 댓글 보기 ({len(quotes)}개)"):
                for g in quotes:
                    st.markdown(f"{ICON[g['dir']]} {md(g['quote'])}  \n"
                                f'<span class="meta">[{g["ref"]}] {html.escape(g["title"])}'
                                f'{"" if g["quote_src"] == "llm" else " · 사례 요약의 대표 댓글로 대체"}</span>',
                                unsafe_allow_html=True)
    if r["sources"]:
        with st.container(key=f"card_sources_{t}"):
            card_h(f"출처 {len(r['sources'])}개")
            with st.expander("목록 보기"):
                for s in r["sources"]:
                    # 가이드 4-6: 번호·제목·발행처·발행일·링크 + 출처 종류, 원문 일부(팝업)·유사도
                    meta = " · ".join(x for x in [s["종류"], s["발행처"], s["발행일"][:10],
                                                  s["쪽"] if s["종류"] == "정부 기준" else ""] if x)
                    st.markdown(f'<p class="src-t">[{s["번호"]}] {html.escape(s["제목"])}</p>'
                                f'<p class="meta">{html.escape(meta)}</p>', unsafe_allow_html=True)
                    with st.container(horizontal=True, key=f"sbtn_{t}_{s['번호']}"):   # 버튼은 정보 줄 아래 따로
                        if s["링크"]:
                            st.link_button("링크", s["링크"], type="tertiary")
                        if st.button("원문", key=f"src_{t}_{s['번호']}", type="tertiary"):
                            source_dialog(s)
    if r.get("dropped") or any(g.get("problems") for g in r.get("case_log", [])):
        with st.expander("검사 기록 (코드가 지우거나 바꾼 것)"):
            for d in r.get("dropped", []):
                st.caption(f"지운 문장 ({d['why']}): {d['line']}")
            for g in r.get("case_log", []):
                if g.get("problems"):
                    st.caption(f"[{g['case']}] {', '.join(g['problems'])} → {g['quote_src']}")


# ---------- 사이드바 ----------

def sidebar():
    with st.sidebar:
        st.markdown("### AI 광고 반응 예상")
        st.caption("국내 AI 광고 사례의 유튜브 댓글 반응과 정부 공식 문서를 근거로 답합니다. 합법·위법은 판단하지 않습니다.")
        if st.button("새 대화", icon=":material/add:", width="stretch"):
            st.session_state.turns = []
            st.rerun()
        st.divider()
        st.markdown("**필터**")                              # 4-7 필터 UI: 고를 때만 검색·판정 조건에 더함
        st.caption("고른 조건의 자료·사례만 근거로 답합니다. 비워 두면 전체 자료를 씁니다.")
        years = st.slider("기간 (발행 연도)", YEARS[0], YEARS[1], YEARS,
                          help=f"끝까지 두면 그쪽은 제한 없음 ({YEARS[0]}년 이전 기사도 포함)")
        kinds = st.multiselect("문서 종류", KINDS, placeholder="전체")
        inds = st.multiselect("업종", industries(), placeholder="전체", help="사례(유튜브 반응)에만 적용. 뉴스·정부 문서는 그대로")
        st.session_state.filters = normalize(years, kinds, inds)


def render_chat():
    warm_up()
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("turns", [])
    sidebar()
    q = st.session_state.get("q_input") or st.session_state.pop("pending", None)   # 입력창 값은 위젯보다 먼저 읽을 수 있음
    has_side = bool(st.session_state.turns or q)            # 첫 답변부터 오른쪽 근거 칸이 나타남
    left, right = st.columns([3, 2], gap="medium") if has_side else (st.container(), None)
    with left:
        if not has_side:
            st.markdown("#### 무엇이 궁금한가요?")
            st.caption("광고 특징(업종·AI를 쓴 부분·AI 사용 공개 여부)을 넣어 반응을 예상하거나, 사례 비교·저작권 기준을 물어보세요.")
        for t in st.session_state.turns:                    # 지난 대화 다시 그리기
            with st.chat_message("user"):
                st.markdown(md(t.get("asked", t["question"])))
            with st.chat_message("assistant"):
                show_main(t)
        if q:
            with st.chat_message("user"):
                st.markdown(md(q))
            with st.chat_message("assistant"):
                try:
                    st.session_state.turns.append(ask(q))
                except Exception as e:                      # 오류를 삼키지 않고 화면에 표시
                    st.error(f"답변 중 오류: {type(e).__name__}: {e}")
        if not has_side:                                    # 첫 화면: 입력창 바로 위 예시 질문 블록
            st.caption("예시 질문 (누르면 바로 물어봅니다)")
            with st.container(horizontal=True, key="exrow"):
                for i, e in enumerate(EXAMPLES):
                    if st.button(e, key=f"ex_{i}"):
                        st.session_state.pending = e
                        st.rerun()                          # 다음 실행 맨 위에서 pending을 질문으로 읽음
        st.chat_input("예: 식품 광고 영상 전체를 AI로 만들고 AI 사용을 밝히면 반응은?", key="q_input")   # 마지막 답변 바로 아래
    if right is None or not st.session_state.turns:
        return
    with right:
        turns = st.session_state.turns
        i = len(turns) - 1
        if len(turns) > 1:                                  # 대화가 늘면 선택지가 바뀌어 기본값이 최근 답변으로 돌아감
            i = st.selectbox("근거를 볼 답변", range(len(turns)), index=i,
                             format_func=lambda n: f"{n + 1}. {turns[n].get('asked', turns[n]['question'])[:30]}")
        show_side(turns[i], i)