"""기획서 확인 화면 (심화, 기획서 9-1). 조회만 하고 인덱스에는 쓰지 않음
- 붙여 넣은 기획서 → rag.plan_check.analyze: AI 사용 항목·특징·등급(코드 규칙) 카드
- 항목을 고르면 그 카드 바로 아래에 펼침: 등급 한 줄(코드) → 가상 소비자 한마디 + 이유 피드백(rag.plan_feedback, 코드 검사) → 근거 사례 자세히(펼침)
  chain.prepare(question, info=…)로 카드와 같은 특징을 넘김 (특징을 다시 뽑지 않아 카드와 답변의 등급이 같음)
- 기획서는 붙여 넣기 또는 파일 올리기(PDF·Word·HWPX, app/file_text.py). 세션에만 두고 저장·임베딩하지 않음. 오른쪽 근거 카드는 챗봇 화면(show_side)과 같은 것
실행: app/main.py의 '기획서 확인' 탭
"""
import html
from pathlib import Path

import streamlit as st

from app.file_text import TYPES, read_upload
from app.chat import CSS as CHAT_CSS, lines, md, show_main, show_side, warm_up
from rag.chain import NOT_FOUND, prepare, stream_body
from rag.plan_check import analyze
from rag.plan_feedback import feedback

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "eval" / "plans" / "plan_C.md"          # 작성자 전시기획안을 고쳐 쓴 더미 (읽기만)
HINT = {"ai_part": "AI로 만드는 부분(인물·제품·배경·영상 전체)", "industry": "회사가 하는 일(업종)",
        "disclosed": "AI 사용 표기 방침", "period": "집행 시기"}   # 빈 특징 → 보강 안내 (코드, LLM 없음)
FEATS = [("ai_part", "AI 부분"), ("industry", "업종"), ("disclosed", "공개"), ("period", "시기")]
CSS = """<style>
[class*="st-key-item_"] {background:#FFFFFF; border:1px solid #E6E8EB; border-radius:14px; padding:0.9rem 1.1rem 0.7rem;}
[class*="st-key-itemsel_"] {background:#FFFFFF; border:1.5px solid #E8877E; border-radius:14px; padding:0.9rem 1.1rem 0.7rem;}
.it-q {font-size:0.98rem; font-weight:600; color:#1F2329; line-height:1.55; margin:0 0 0.5rem;}
.chip {display:inline-block; font-size:0.78rem; color:#374151; background:#F3F1EE; border-radius:999px; padding:0.1rem 0.6rem; margin:0 0.3rem 0.3rem 0;}
.chip.none {color:#9CA3AF; background:#F7F6F4;}
.grade {display:inline-block; font-size:0.8rem; font-weight:700; border-radius:6px; padding:0.12rem 0.55rem; margin:0.2rem 0.4rem 0 0;}
.grade.obs {background:#F8DDD9; color:#B85C54;}
.grade.weak {background:#EDEAE6; color:#6B7280;}
.persona {background:#FDF3F1; border-left:3px solid #E8877E; border-radius:10px; padding:0.7rem 0.9rem; margin:0.6rem 0 0.8rem;}
.persona .who {font-size:0.78rem; color:#B85C54; margin:0 0 0.2rem;}
.persona .say {font-size:1.02rem; font-weight:700; color:#1F2329; line-height:1.5; margin:0;}
.persona .by {font-size:0.82rem; color:#6B7280; margin:0.25rem 0 0;}
.persona.pos {background:#EAF4EE; border-left-color:#5E9E76;}   /* 긍정: 초록 (부정은 코랄). 아이콘 👍/👎으로도 구분 */
.persona.pos .who {color:#3F7A55;}
.fb-h {font-size:0.9rem; font-weight:700; color:#1F2329; margin:0.4rem 0 0.1rem;}
.hint {font-size:0.82rem; color:#6B7280; margin:0.5rem 0 0; padding-left:0.6rem; border-left:2px solid #D6D1CB;}
.lead {font-size:1.15rem; font-weight:700; color:#1F2329; margin:1.4rem 0 0.1rem;}
</style>"""


def _load_example():
    st.session_state.plan_text = EXAMPLE.read_text(encoding="utf-8")


def _load_file():
    """파일을 올리면 글자를 꺼내 입력 칸에 넣음 (사용자가 확인·수정한 뒤 확인 버튼)"""
    f = st.session_state.get("plan_file")
    if f is None:
        return
    try:
        text, note = read_upload(f.name, f.getvalue())
    except Exception as e:                               # 오류는 삼키지 않고 화면·터미널에 남김
        print(f"[file_text 오류] {f.name}: {type(e).__name__}: {e}")
        st.session_state.plan_note = ("error", f"파일을 읽는 중 오류가 났습니다 ({type(e).__name__}). 내용을 직접 붙여 넣어 주세요.")
        return
    if text:
        st.session_state.plan_text = text
    st.session_state.plan_note = ("warning" if not text or "띄어쓰기" in note else "info", note)


def _card_html(it):
    chips = "".join(f'<span class="chip">{name} {html.escape(it["features"][k])}</span>' if it["features"][k]
                    else f'<span class="chip none">{name} 기획서에 없음</span>' for k, name in FEATS)
    n = len(it["similar"])
    grade = (f'<span class="grade obs">● 관찰됨</span>' if it["grade"] == "관찰됨"
             else f'<span class="grade weak">○ 근거 부족</span>')   # 색만으로 구분하지 않게 기호·글자도
    miss = [HINT[k] for k, _ in FEATS if not it["features"][k]]
    hint = (f'<p class="hint">비슷한 사례를 더 정확히 찾으려면 기획서에 적어 주세요: {", ".join(miss)}</p>' if miss else "")
    return (f'<p class="it-q">{html.escape(it["quote"])}</p>{chips}<br>{grade}'
            f'<span class="chip none" style="background:none;padding-left:0">비슷한 사례 {n}개</span>{hint}')


def _lead(res):
    """해석 제목: 숫자는 결과에서 계산"""
    n, k = len(res), sum(it["grade"] == "관찰됨" for it in res)
    if n == 0:
        return "AI를 쓰는 항목을 찾지 못했습니다"
    if k == 0:
        return f"AI를 쓰는 항목 {n}개 모두 비슷한 사례가 2개 미만이라 반응 근거가 부족합니다"
    return f"AI를 쓰는 항목 {n}개 중 {k}개는 비슷한 사례가 2개 이상입니다"


def _answer(it):
    """준비(prepare) → 사례별 댓글 고르기(stream_body, 화면에 흘리지 않고 모음) → 피드백. 진행 상태만 보여 줌"""
    with st.status("답변 준비 중", expanded=False) as status:
        p = prepare(it["question"], progress=lambda m: status.update(label=m), info=it["info"])
        out = {}
        if not p["refusal"]:
            status.update(label="③ 사례별 실제 댓글 고르는 중")
            for _ in stream_body(p, out):
                pass
        r, fb = {**p, **out}, None
        try:
            status.update(label="④ 가상 소비자 한마디와 이유 정리 중")
            fb = feedback(it, r)
        except Exception as e:                           # 피드백이 실패해도 근거 답변은 보여 줌 (오류는 기록)
            print(f"[plan_feedback 오류] {type(e).__name__}: {e}")
            r["fb_error"] = f"{type(e).__name__}"
        status.update(label="자료 확인 끝", state="complete")
    return {**r, "fb": fb}


def _show_answer(r):
    """카드 안에 펼치는 답변: 등급 한 줄 → 한마디·피드백 → 근거 사례 자세히(펼침) → 주의"""
    if r["refusal"]:
        show_main(r)
        return
    head = (r["head"] or "").split("\n")
    if head[0]:
        st.markdown(lines(head[0]))
    fb = r.get("fb")
    if fb:
        if fb["line"]:
            icon, tone = ("👎", "") if fb["dir"] == "부정" else ("👍", " pos")
            st.markdown(f'<div class="persona{tone}"><p class="who">가상의 소비자 한마디 (실제 댓글 아님)</p>'
                        f'<p class="say">{icon} “{html.escape(fb["line"])}”</p>'
                        f'<p class="by">- {html.escape(fb["persona"])}</p></div>', unsafe_allow_html=True)
            st.caption(md(fb["dir_note"]))
        if fb["feedback"]:
            st.markdown('<p class="fb-h">이런 반응이 나올 이유</p>', unsafe_allow_html=True)
            st.markdown("\n".join(f"- {md(t)}" for t in fb["feedback"]))
        if not fb["line"] and not fb["feedback"]:
            st.caption("근거와 맞지 않는 문장을 지우고 나니 피드백이 남지 않았습니다. 아래 근거 사례를 확인해 주세요.")
    elif r.get("fb_error"):
        st.caption(f"피드백을 만들지 못했습니다 ({r['fb_error']}). 아래 근거 사례를 확인해 주세요.")
    elif r.get("is_predict"):
        st.caption("비슷한 사례가 2개 미만이라 가상 소비자 한마디는 만들지 않았습니다. 사례 하나로 반응을 그려 내면 과장이 되기 때문입니다.")
    rest = "\n".join(head[1:]).strip()
    if rest or r.get("body"):
        with st.expander("근거 사례 자세히", expanded=not fb):
            if rest:
                st.markdown(lines(rest))
            if r.get("body") and r["body"] != NOT_FOUND:
                st.markdown(lines(r["body"]))
    if r["foot"]:
        st.caption(md(r["foot"]))
    if fb and fb["dropped"]:
        with st.expander("검사 기록 (코드가 지운 피드백 문장)"):
            for d in fb["dropped"]:
                st.caption(md(d))


def render_plan():
    warm_up()
    st.markdown(CHAT_CSS + CSS, unsafe_allow_html=True)
    ss = st.session_state
    for k, v in {"plan_text": "", "plan_res": None, "plan_src": "", "plan_sel": None, "plan_ans": {}}.items():
        ss.setdefault(k, v)

    st.markdown("#### 기획서 확인")
    st.caption("기획서를 붙여 넣으면 AI를 쓰는 항목을 찾아, 항목마다 비슷한 국내 사례의 댓글 반응을 근거로 보여 드립니다. "
               "예상 등급은 코드 규칙으로 정하며, 올리거나 붙여 넣은 기획서는 저장하지 않습니다.")
    st.file_uploader("기획서 파일 올리기 (PDF · Word · 한글 HWPX)", type=TYPES, key="plan_file", on_change=_load_file,
                     help="파일은 화면에서 글자만 읽고 저장하지 않습니다. 구형 한글(.hwp)은 PDF나 HWPX로 저장해 올려 주세요.")
    kind, note = ss.get("plan_note") or ("", "")
    if note:
        {"info": st.caption, "warning": st.warning, "error": st.error}[kind](note)
    text = st.text_area("기획서", key="plan_text", height=260, label_visibility="collapsed",
                        placeholder="파일을 올리거나 기획서 내용을 붙여 넣어 주세요. 개요(회사·시기)와 AI 사용 표기 방침이 함께 있으면 더 정확합니다.")
    with st.container(horizontal=True):
        run = st.button("AI 사용 항목 확인", type="primary", disabled=not text.strip())
        st.button("예시 기획서 넣기", type="tertiary", on_click=_load_example)
    if run:
        with st.spinner("항목을 찾고 특징을 읽는 중 (20초 안팎)"):
            try:
                ss.plan_res, ss.plan_src, ss.plan_sel, ss.plan_ans = analyze(text), text, None, {}
            except Exception as e:                       # 오류는 삼키지 않고 화면·터미널에 남김
                print(f"[plan_check 오류] {type(e).__name__}: {e}")
                st.error(f"기획서를 읽는 중 오류가 났습니다: {type(e).__name__}", icon=":material/error:")
                return
    res = ss.plan_res
    if res is None:
        return
    if text != ss.plan_src:
        st.caption("기획서가 바뀌었습니다. 다시 확인하려면 버튼을 눌러 주세요. 아래는 이전 기획서 결과입니다.")

    st.markdown(f'<p class="lead">{html.escape(_lead(res))}</p>', unsafe_allow_html=True)
    st.caption("‘예상 답변 보기’를 누르면 그 카드 아래에 가상 소비자 한마디와 이유, 근거가 펼쳐집니다. 시기·공개 여부는 기획서 개요와 방침에서 읽습니다.")
    for i, it in enumerate(res, 1):
        sel = ss.plan_sel == i
        with st.container(key=f"{'itemsel' if sel else 'item'}_{i}"):
            st.markdown(_card_html(it), unsafe_allow_html=True)
            if st.button("접기" if sel else "예상 답변 보기", key=f"pick_{i}", type="tertiary",
                         icon=":material/expand_less:" if sel else ":material/expand_more:"):
                ss.plan_sel = None if sel else i
                st.rerun()
            if not sel:
                continue
            with st.expander("코드가 조립한 질문 보기"):
                st.text(it["question"])
            left, right = st.columns([3, 2], gap="large")
            with left:
                if i not in ss.plan_ans:
                    ss.plan_ans[i] = _answer(it)
                _show_answer(ss.plan_ans[i])
            with right:
                show_side(ss.plan_ans[i], f"plan{i}")