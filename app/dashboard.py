"""성능 평가 대시보드 (가이드 4-8, 기획서 7-2). eval/results·채점표·Chroma 메타·docs.db를 읽기만 한다 (임베딩·쓰기 없음)
- 섹션마다: 작은 번호 줄 | n. 이름 | → 해석 제목 → 설명. 문장 속 숫자는 결과 파일에서 계산
- 원포인트 컬러(파스텔): 제목이 말하는 부분만 포인트 색(ACCENT, 챗봇 #FF4B4B의 파스텔), 반대 무리는 대비색(COMP), 나머지는 회색
실행: app/main.py의 '성능 평가' 탭
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag.prompts import REASON_NAME                     # 이유 코드 → 이름 (상수만, LLM 없음)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
def res(name):
    """eval/results 아래 하위 폴더까지 찾아 파일 경로를 돌려줌 (폴더로 묶은 뒤에도 같은 이름으로 읽기)"""
    hits = sorted(RES.rglob(name))
    return hits[0] if hits else RES / name

EXPS = [("기준선", "run1_chain"), ("exp9", "exp9_feat_prompt_r1"), ("exp10 (최종)", "exp10_source_quota")]
GRADES = [("개선 전", "day5_grading.csv"), ("개선 후", "final_grading.csv")]
TYPE_KO = {"predict": "예상", "fact_reaction": "사실·반응", "compare": "비교", "fact_policy": "사실·정책", "synthesis": "종합"}
# A안: 웜 그레이 + 코랄 (챗봇 #FF4B4B 계열과 온도를 맞춤). 코랄 ↔ 파랑은 밝기도 달라(L 67 / 58) 색약·흑백에서도 구분됨
ACCENT = "#E8877E"                                      # 포인트(코랄)
ACCENT_LIGHT, ACCENT_DEEP = "#F8DDD9", "#B85C54"        # 강조 배경 / 강조 글자
COMP = "#6E8DB5"                                        # 대비색(차분한 파랑)
GRAYS = ["#EDEAE6", "#D6D1CB", "#B3ABA3"]               # 웜 그레이, 옅은 → 진한
KIND_COLOR = {"소비자 반응": ACCENT, "언론 보도": ACCENT_LIGHT, "정부 기준": ACCENT_DEEP}   # 코랄 계열 밝기 3단계
FONT = dict(family="Pretendard, sans-serif", size=13, color="#374151")
CSS = """<style>
.sec-idx {font-size:0.85rem; color:#6B7280; margin:0 0 -0.5rem; letter-spacing:0.02em;}
[class*="st-key-sec_"] {background:#FFFFFF; border:1px solid #E6E8EB; border-radius:16px; padding:1.4rem 1.6rem 0.6rem; margin-top:1.6rem;}
.lgd {font-size:0.85rem; color:#374151; margin:0.2rem 0 -0.6rem;}
.lgd span {display:inline-block; width:0.8rem; height:0.8rem; border-radius:2px; margin:0 0.3rem -0.1rem 0.9rem;}
.facet-t {text-align:center; font-weight:600; color:#374151; margin:0.8rem 0 -0.8rem;}
</style>"""
INTRO_STAGE = """| 구분 | 내용 |
|---|---|
| 개선 전 | 1차 완성본. 지식베이스(유튜브 댓글 2,290개 분류, 검색 단위 564개), 검색·답변 흐름, 예상 등급을 정하는 코드 규칙까지 만든 상태 |
| 개선 후 | 아래 실험 세 가지(exp9·exp13·exp10)를 모두 적용한 최종본 |"""
INTRO_EXP = """| 실험 | 바꾼 것 |
|---|---|
| exp9 | 질문에서 광고 특징을 뽑을 때 ‘AI를 쓴 부분’(모델 인물·제품 이미지·배경 일부·영상 전체)을 고르는 기준을 보강 |
| exp13 | 질문에 사례 이름이 있으면, 답 맨 앞에 코드가 그 사례의 댓글 수치(부정·긍정 비율, 주된 이유)를 붙임 |
| exp10 | 반응과 정부 기준을 함께 묻는 종합 질문은 사례 댓글과 정부 문서를 나눠 찾아 둘 다 근거로 씀 |"""


def _style(fig, h=380, legend=True):
    fig.update_layout(template="plotly_white", font=FONT, height=h, margin=dict(l=10, r=10, t=40 if legend else 10, b=10),
                      showlegend=legend, bargap=0.25,
                      legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.02, x=0))   # 범례는 차트 위 가로로
    return fig


def _show(fig):
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except TypeError:                                   # 옛 Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def section(key, idx, headline, caption):
    """흰 카드 하나 = 섹션 하나. 작은 번호 줄 → 해석 제목 → 설명 (~ 는 취소선이 되므로 쓰지 않음)"""
    box = st.container(key=f"sec_{key}")
    with box:
        st.markdown(f'<p class="sec-idx">| {idx} |</p>', unsafe_allow_html=True)
        st.subheader(headline)
        st.caption(caption)
    return box


# ---------- 데이터 (모두 읽기만) ----------

@st.cache_data
def load_runs(exps=tuple(EXPS)):
    rows = []
    for name, tag in exps:
        p = res(f"{tag}.json")
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            rows += [{"실험": name, "지표": k, "값": s[v]} for k, v in [("Hit@1", "hit1"), ("Hit@3", "hit3"), ("MRR", "mrr10")]]
    return pd.DataFrame(rows)


@st.cache_data
def load_grades(grades=tuple(GRADES)):
    out = [pd.read_csv(ROOT / "eval" / "grading" / f, encoding="utf-8-sig", dtype=str).fillna("").assign(시점=w)
           for w, f in grades if (ROOT / "eval" / "grading" / f).exists()]
    return pd.concat(out) if out else pd.DataFrame()


@st.cache_data
def load_scores():
    p = res(f"{EXPS[-1][1]}_detail.csv")
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, encoding="utf-8-sig")
    d = d[d.type != "unanswerable"].copy()
    d["점수"] = d.top3.str.extract(r"\(([\d.]+)\)")[0].astype(float)   # top3 첫 칸 = 1위 유사도
    d["결과"] = d.hit1.map({1: "1위가 정답", 0: "1위가 오답"})
    return d


@st.cache_data
def load_meta():
    from rag.retriever import _col                      # 메타데이터만 조회 (임베딩 모델은 불러오지 않음)
    return pd.DataFrame(_col().get(include=["metadatas"])["metadatas"])


def case_names():
    m = load_meta()
    m = m[m["doc_type"] == "case_summary"] if "doc_type" in m else m.iloc[0:0]
    pick = lambda r: next((r[c] for c in ["display_name", "title"] if c in r and isinstance(r[c], str) and r[c]), None)
    return {r["case_id"]: pick(r) for _, r in m.iterrows()}


@st.cache_data
def load_cases():
    con = sqlite3.connect(f"file:{ROOT / 'store' / 'docs.db'}?mode=ro", uri=True)
    df = pd.read_sql("select case_id, stats from case_summary", con)
    con.close()
    try:
        names = case_names()
    except Exception:                                   # 이름을 못 읽어도 case_id로 표시
        names = {}

    def reason(m):
        if not m:
            return "없음"
        return REASON_NAME.get(m["code"], m["code"]) + (f" · {REASON_NAME.get(m['second']['code'], '')} (동률)" if m.get("tie") else "")

    rows = []
    for r in df.itertuples():
        s = json.loads(r.stats)
        rows.append({"사례": (names.get(r.case_id) or r.case_id) + ("" if s["qualified"] else " (표본 부족)"),
                     "부정": s["neg_pct"], "긍정": s["pos_pct"], "자격": s["qualified"], "관련 댓글": s["top_related"],
                     "부정 이유": reason(s["neg_main"]), "긍정 이유": reason(s["pos_main"])})
    return pd.DataFrame(rows).sort_values("부정")


def n_grade(g, when, t, grade):
    return int(((g.시점 == when) & (g.type == t) & (g.정확도 == grade)).sum())


# ---------- 화면 ----------

def header(g, runs):
    st.title("챗봇 성능 평가")
    st.markdown("챗봇의 **검색 정확도**와 **답변 품질**을, 40문항으로 구성한 평가셋(답변하면 안 되는 질문 8문항 포함)으로 "
                "개선 전과 후로 나누어 측정했습니다.")
    with st.expander("개선 전·후와 실험(exp) 설명", expanded=True):
        st.markdown("**개선 전·후**")
        st.markdown(INTRO_STAGE)
        st.markdown("**실험(exp) 설명**")
        st.markdown(INTRO_EXP)
        st.caption("exp 번호는 실패 사례를 분석하며 붙인 개선 후보 번호입니다. 한 번에 하나씩만 바꿔 효과를 따로 쟀습니다.")
    c = st.columns(4)
    if not runs.empty:
        h3 = runs[runs.지표 == "Hit@3"].값.tolist()
        c[0].metric("Hit@3 (최종)", f"{h3[-1]:.3f}", f"{h3[-1] - h3[0]:+.3f}")
    if g.empty:
        return
    acc = {w: (x.정확도 == "정확").mean() * 100 for w, x in g[g.type != "unanswerable"].groupby("시점")}
    if "개선 후" in acc:
        c[1].metric("답변 정확률 (32문항)", f"{acc['개선 후']:.1f}%", f"{acc['개선 후'] - acc.get('개선 전', acc['개선 후']):+.1f}%p")
    after = g[g.시점 == "개선 후"]
    u = after[after.type == "unanswerable"]
    c[2].metric("답변 불가 질문 거절", f"{u.거절.str.startswith('성공').sum()}/{len(u)}")
    src = after.출처정확성
    c[3].metric("출처 정확", f"{(src == 'O').sum()}/{src.isin(['O', 'X']).sum()}", help="'확인 필요'·'해당없음'은 제외했습니다")


def sec_search(runs):
    h3 = runs[runs.지표 == "Hit@3"].값.tolist()
    with section("search", "1. 실험(exp)별 검색 성능", f"종합 질문의 자료를 출처별로 나눠 찾자 Hit@3가 {h3[0]:.3f} → {h3[-1]:.3f}로 올랐습니다",
                 f"Hit@3는 정답 자료가 검색 결과 상위 3개 안에 든 문항의 비율입니다(32문항 중 {round(h3[0] * 32)}개 → {round(h3[-1] * 32)}개). "
                 "Hit@1은 1위 자료가 정답인 비율, MRR은 정답이 몇 위에 나왔는지를 점수로 바꾼 평균입니다. "
                 "기준선(파랑)과 최종(빨강)을 비교해 보세요. exp9는 검색 점수에 변화가 없었고 exp10에서 올랐습니다. exp13은 검색을 바꾸지 않아 표에 넣지 않았습니다."):
        fig = px.bar(runs, x="지표", y="값", color="실험", barmode="group", text="값",
                     color_discrete_sequence=[COMP, GRAYS[0], ACCENT])   # 기준선 ↔ 최종을 대비색으로
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        _show(_style(fig).update_yaxes(range=[0, 1.08], showgrid=False, visible=False).update_xaxes(title=None))


def sec_grades(g):
    s = lambda w, k: n_grade(g, w, "synthesis", k)
    with section("grades", "2. 질문 유형별 답변 정확도", "‘예상·비교’ 질문에는 강하지만, ‘종합’ 질문에서는 아직 약합니다",
                 "예상·비교는 코드가 등급과 수치를 정하고 LLM은 그 안에서 문장만 쓰는 방식이라 정확도가 높습니다. "
                 f"LLM이 스스로 내용을 엮어야 하는 종합 질문은 개선 전 6문항 중 오답 {s('개선 전', '오답')}개·부분 {s('개선 전', '부분')}개였고, "
                 f"exp10 뒤 오답 {s('개선 후', '오답')}개·부분 {s('개선 후', '부분')}개로 나아졌지만 아직 정확은 {s('개선 후', '정확')}개입니다. "
                 f"사실·반응 질문은 exp13으로 정확이 {n_grade(g, '개선 전', 'fact_reaction', '정확')}개에서 "
                 f"{n_grade(g, '개선 후', 'fact_reaction', '정확')}개가 됐습니다."):
        st.caption("판정 기준 — 정확: 정답 기준을 모두 충족 / 부분: 대상(사례·문서)은 맞게 찾았고 내용 일부가 맞음 / "
                   "오답: 대상을 못 찾았거나 질문을 잘못 읽었거나 근거와 반대되는 내용")
        a = g[g.type != "unanswerable"]
        cats = list(TYPE_KO.values())
        st.markdown('<p class="lgd">' + "".join(f'<span style="background:{c}"></span>{t}' for t, c in
                    [("정확", GRAYS[2]), ("부분", GRAYS[1]), ("오답", GRAYS[0])]) + "</p>", unsafe_allow_html=True)
        cols = st.columns(2, gap="large")
        for col, when in zip(cols, ["개선 전", "개선 후"]):
            fig = go.Figure()
            for j, grade in enumerate(["정확", "부분", "오답"]):          # 짙을수록 좋음: 정확 > 부분 > 오답
                y = [n_grade(a, when, t, grade) for t in TYPE_KO]
                fig.add_bar(x=cats, y=y, name=grade, marker_color=GRAYS[2 - j],      # 판정 색은 모든 유형이 같음 text=[v or "" for v in y],
                            textposition="inside", insidetextanchor="middle", textfont=dict(color="#374151"))
            k = cats.index("종합")                     # 종합 열만 배경 강조 + 위에 오답 수 표시
            fig.add_vrect(x0=k - 0.5, x1=k + 0.5, fillcolor=ACCENT_LIGHT, opacity=1, layer="below", line_width=0)
            fig.add_annotation(x="종합", y=n_grade(a, when, "synthesis", "정확") + n_grade(a, when, "synthesis", "부분")
                               + n_grade(a, when, "synthesis", "오답"), text=f"<b>오답 {n_grade(a, when, 'synthesis', '오답')}개</b>",
                               showarrow=False, yshift=14, font=dict(color=ACCENT_DEEP, size=13))
            fig.update_layout(barmode="stack")
            fig.update_xaxes(tickvals=cats, ticktext=[f"<b>{c}</b>" if c == "종합" else c for c in cats])
            with col:
                st.markdown(f'<p class="facet-t">{when}</p>', unsafe_allow_html=True)
                _show(_style(fig, h=340, legend=False).update_yaxes(range=[0, 11.2], dtick=2, title=None))
        st.caption("막대는 아래부터 정확(가장 짙음) → 부분 → 오답(가장 옅음)이고, 모든 유형이 같은 색입니다. 제목이 말하는 ‘종합’ 열만 배경을 칠하고 오답 수를 위에 적었습니다.")


def sec_scores(sc):
    ok, ng = sc[sc.hit1 == 1].점수, sc[sc.hit1 == 0].점수
    k = int((ok <= ng.max()).sum())
    with section("scores", "3. 검색 1위 유사도 분포", "검색 점수만으로는 ‘제대로 찾았는지’ 가릴 수 없었습니다",
                 "유사도는 질문과 검색된 자료의 뜻이 얼마나 가까운지를 0–1로 나타낸 점수입니다. "
                 f"1위 자료가 정답이었던 문항은 {ok.min():.2f}–{ok.max():.2f}, 오답이었던 문항은 {ng.min():.2f}–{ng.max():.2f}로 "
                 "범위가 겹칩니다(색칠한 띠). "
                 f"오답 {len(ng)}개를 모두 거르려면 ‘{ng.max():.2f} 이하면 자료 없음’처럼 기준을 정해야 하는데, 그러면 정답 문항 {len(ok)}개 중 "
                 f"{k}개도 함께 걸러집니다. 그래서 ‘자료 없음’은 점수 기준 대신 질문 범위 판정과 답변 단계의 "
                 "‘확인할 수 없음’ 처리로 거릅니다."):
        fig = go.Figure()
        fig.add_hrect(y0=ng.min(), y1=ng.max(), fillcolor=ACCENT_LIGHT, opacity=0.8, line_width=0, layer="below",   # 띠는 상자 뒤에
                      annotation_text="겹치는 구간", annotation_position="top right", annotation_font_color=ACCENT_DEEP)
        for name, part, sym in [("1위가 정답", sc[sc.hit1 == 1], "circle"), ("1위가 오답", sc[sc.hit1 == 0], "diamond")]:
            fig.add_box(y=part.점수, name=f"{name} ({len(part)}문항)", boxpoints="all", jitter=0.4, pointpos=0,
                        marker=dict(color="#8A817A", size=8, symbol=sym), line=dict(color="#6F6761", width=2),
                        fillcolor="rgba(255,255,255,0.55)", customdata=part.id,
                        hovertemplate="%{customdata} · 유사도 %{y:.3f}<extra></extra>")
        _show(_style(fig, legend=False).update_yaxes(title="1위 유사도"))


def sec_kb():
    m = load_meta()
    m = m.assign(종류=m["authority"].fillna("기타"), 연도=m["year"].fillna(0).astype(int))
    yrs = lambda k: sorted(m[(m.종류 == k) & (m.연도 > 0)].연도.unique())
    rng = lambda k: f"{yrs(k)[0]}–{yrs(k)[-1]}년" if yrs(k) else "연도 없음"
    n = m.종류.value_counts()
    kinds = [k for k in ["소비자 반응", "언론 보도", "정부 기준"] if k in n]
    with section("kb", "4. 지식베이스 구성 (검색 단위 = 청크 기준)",
                 f"검색 단위 총 {len(m)}개 - {'·'.join(kinds)} {len(kinds)}종 구성",
                 " / ".join(f"{k} {n[k]}개" for k in kinds) + ". "
                 f"정부 기준은 {'·'.join(map(str, yrs('정부 기준')))}년 발행본뿐이고, 소비자 반응은 {rng('소비자 반응')}, "
                 f"언론 보도는 {rng('언론 보도')}에 걸쳐 있습니다. 정부 문서가 나온 뒤의 사례도 있어서, 답변에는 정부 자료의 발행일을 함께 표시합니다."):
        kb = m.groupby(["연도", "종류"]).size().reset_index(name="개수")
        kb["연도"] = kb.연도.astype(str).replace("0", "연도 없음")
        fig = go.Figure()
        for k in kinds:                                 # 붉은 계열 밝기로 종류 구분
            d = kb[kb.종류 == k]
            fig.add_bar(x=d.연도, y=d.개수, name=f"{k} ({n[k]})", marker_color=KIND_COLOR[k])
        fig.update_layout(barmode="stack")
        _show(_style(fig).update_xaxes(type="category", categoryorder="category ascending").update_yaxes(title="검색 단위 수"))


def sec_cases(cs):
    q = cs[cs.자격 == 1]
    high, low = q[q.부정 >= 50], q[q.부정 < 50]           # 부정 댓글이 절반 이상인 무리 / 미만인 무리
    lgd = [("부정 50% 이상", ACCENT), ("부정 50% 미만", COMP), ("표본 부족", GRAYS[0])]
    with section("cases", "5. 사례별 유튜브 댓글 반응", "똑같이 AI를 썼어도 소비자 반응은 정반대로 갈렸습니다",
                 f"댓글 표본이 충분한 사례 {len(q)}개가 부정 비율 {low.부정.max()}% 이하(파랑) {len(low)}개와 "
                 f"{high.부정.min()}% 이상(빨강) {len(high)}개, 두 무리로 뚜렷이 나뉩니다(최상위 댓글 기준). "
                 "그래서 이 챗봇은 ‘AI 광고 전반’이 아니라 특징이 비슷한 사례끼리 묶어 반응을 예상합니다. "
                 "‘표본 부족’(회색)은 관련 댓글이 적어 예상·비교 근거로 쓰지 않는 사례입니다. 막대에 마우스를 올리면 긍정 비율과 주된 이유가 보입니다."):
        st.markdown('<p class="lgd">' + "".join(f'<span style="background:{c}"></span>{t}' for t, c in lgd) + "</p>",
                    unsafe_allow_html=True)
        colors = [(ACCENT if r.부정 >= 50 else COMP) if r.자격 else GRAYS[0] for r in cs.itertuples()]
        fig = go.Figure(go.Bar(
            y=cs.사례, x=cs.부정, orientation="h", marker_color=colors, text=[f"{v}%" for v in cs.부정], textposition="outside",
            customdata=cs[["긍정", "관련 댓글", "부정 이유", "긍정 이유"]].values,
            hovertemplate="<b>%{y}</b><br>부정 %{x}% · 긍정 %{customdata[0]}%<br>관련 댓글 %{customdata[1]}개"
                          "<br>부정 주된 이유: %{customdata[2]}<br>긍정 주된 이유: %{customdata[3]}<extra></extra>"))
        _show(_style(fig, h=40 * len(cs) + 60, legend=False).update_xaxes(title="부정 댓글 비율(%)", range=[0, 100]).update_yaxes(title=None))


def render_dashboard():
    st.markdown(CSS, unsafe_allow_html=True)
    runs, g = load_runs(), load_grades()
    header(g, runs)
    if not runs.empty:
        sec_search(runs)
    if not g.empty:
        sec_grades(g)
    sc = load_scores()
    if not sc.empty and (sc.hit1 == 0).any():
        sec_scores(sc)
    for f, name in [(sec_kb, "지식베이스 메타데이터"), (lambda: sec_cases(load_cases()), "사례 요약")]:
        try:
            f()
        except Exception as e:                          # 화면은 계속 보이게, 원인은 표시
            st.warning(f"{name}를 읽지 못했습니다: {type(e).__name__}: {e}")