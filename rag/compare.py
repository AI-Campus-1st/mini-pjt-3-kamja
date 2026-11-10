"""
비교 질문 처리 (Day 4 추가). 비교에 필요한 수치는 코드가 이미 안다: 사례 특징(사례 요약 메타) + 비율·주된 이유(case_summary.stats)
- detect(q): 비교 질문인지, 특징 값끼리인지(by) 이름을 댄 사례끼리인지(case_ids) LLM이 판정 (특징 추출과 별도 호출)
- build(plan): 코드가 비교 표를 문장으로 만듦. 자격 사례(qualified=1)만 근거, 사례 2개 미만인 값은 "관찰 수준" 문장을 코드가 붙임
- LLM은 표를 보충하는 설명만 쓰고, 수치·일반화 문장은 chain.py가 지움
"""

from functools import lru_cache

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.prompts import FEAT_NAME, REASON_NAME
from rag.reaction import FEATS, cases
from rag.retriever import get_summaries

MODEL = "gpt-4.1-mini"


@lru_cache(maxsize=1)
def titles():
    """case_id → 사례명 (사례 요약 청크 메타의 title, 읽기만)"""
    return {d["meta"]["case_id"]: d["meta"].get("title", d["meta"]["case_id"]) for d in get_summaries(sorted(cases()))}


@lru_cache(maxsize=1)
def names():
    """case_id → 화면 표시용 사례명 (display_name, 없으면 title). 판정(detect)은 원래 title을 씀"""
    return {d["meta"]["case_id"]: d["meta"].get("display_name") or d["meta"].get("title", d["meta"]["case_id"])
            for d in get_summaries(sorted(cases()))}


def _prompt():
    lines = "\n".join(f"- {cid}: {titles()[cid]}" for cid in sorted(cases()))
    feats = ", ".join(f"{f}({FEAT_NAME[f]})" for f in FEATS)
    return f"""너는 질문이 사례 반응을 비교하는 질문인지 판정한다. JSON 객체 하나만 출력한다.
사례 목록
{lines}
- compare: 두 개 이상의 사례나 광고 특징 값의 소비자 반응을 비교·대조하는 질문이면 true
- case_ids: 질문이 이름(브랜드·광고명)으로 가리킨 사례의 case_id 목록. 없으면 []
- by: 사례 이름 없이 광고 특징 값끼리 비교하면 그 특징 이름 ({feats} 중 하나), 아니면 null
  (예: AI 사용을 밝힌 것과 밝히지 않은 것 → disclosed, 일부만 AI와 전체 AI → ai_part)
출력 형식: {{{{"compare": true, "case_ids": [], "by": null}}}}"""


@lru_cache(maxsize=1)
def _chain():
    llm = ChatOpenAI(model=MODEL, temperature=0, default_headers={"Accept-Encoding": "gzip, deflate"}) \
        .bind(response_format={"type": "json_object"})   # 키는 .env (chain.py가 features.py를 통해 읽음)
    return ChatPromptTemplate.from_messages([("system", _prompt()), ("human", "{q}")]) | llm | JsonOutputParser()


def named_cases(q):
    """질문에 사례명 첫 단어(브랜드)가 있으면 그 사례 (코드 대조, LLM 흔들림 보완)"""
    return [cid for cid, t in sorted(titles().items()) if t.split()[0] in q]


def detect(q):
    """→ {"mode": "cases"|"feature"|None, "case_ids": [...], "by": 특징 또는 None, "by_name": 코드 대조로 잡았는지, "error": ...}
    이름 댄 사례 2개 이상이 우선. LLM이 비교가 아니라고 해도 질문에 사례 이름이 2개 이상 있으면 사례 비교로 봄"""
    try:
        raw, err = _chain().invoke({"q": q}), None
    except Exception as e:
        raw, err = {}, f"{type(e).__name__}: {e}"[:300]
        print(f"[compare 오류] {err}")
    names = named_cases(q)
    ids = [c for c in dict.fromkeys(raw.get("case_ids") or []) if c in cases()] if raw.get("compare") else []
    by = raw.get("by") if raw.get("compare") and raw.get("by") in FEATS else None
    by_name = len(ids) < 2 and len(names) >= 2
    ids = names if by_name else ids
    mode = "cases" if len(ids) >= 2 else "feature" if by else None
    return {"mode": mode, "case_ids": ids if mode == "cases" else [], "by": by, "by_name": by_name, "error": err}


def _rname(m):
    """주된 이유 이름. 1·2위 동률(tie)이면 두 이유를 함께 (fix_row_tie)"""
    if not m:
        return "-"
    return f'{REASON_NAME[m["code"]]}·{REASON_NAME[m["second"]["code"]]} 동률' if m.get("tie") else REASON_NAME[m["code"]]


def _row(cid, n):
    st = cases()[cid]["stats"]
    return (f"{names()[cid]} [{n}]: 부정 {st['neg_pct']}% · 긍정 {st['pos_pct']}% (댓글 {st['top_related']}개), "
            f"부정 주된 이유 {_rname(st['neg_main'])}, 긍정 주된 이유 {_rname(st['pos_main'])}")


def fact_head(cid):
    """exp13_fact_head: 사례 이름이 하나인 사실 질문 → 코드가 쓰는 그 사례 수치 줄 ([1] = 그 사례 요약)"""
    st = cases()[cid]["stats"]
    if cases()[cid]["qualified"]:
        return f"[사례 수치] 최상위 댓글 기준\n- {_row(cid, 1)}"
    return (f"[사례 수치] 참고용 — 관련 최상위 댓글이 {st['top_related']}개뿐이라 표본이 부족해 예상·비교 근거로 쓰지 않는 사례입니다.\n"
            f"- {_row(cid, 1)}")


def build(plan, allowed=None):
    """비교 표 → {"head": 코드가 쓴 표 문장, "case_ids": 출처 순서, "rows": UI 표용}. 근거로 쓸 사례가 2개 미만이면 None"""
    c = cases()
    nums = {}
    if plan["mode"] == "cases":
        ids = plan["case_ids"]
        ids = [i for i in ids if allowed is None or i in allowed]          # 필터 밖 사례는 비교에서 뺌
        ok, bad = [i for i in ids if c[i]["qualified"]], [i for i in ids if not c[i]["qualified"]]
        if len(ok) < 2:
            return None
        lines = ["[비교] 질문한 사례의 댓글 반응 (최상위 댓글 기준)"] + [f"- {_row(i, n)}" for n, i in enumerate(ok, 1)]
        if bad:
            lines.append(f"(댓글 표본이 부족해 비교 근거로 쓰지 않은 사례: {', '.join(names()[i] for i in bad)})")
        order = ok
    else:
        f = plan["by"]
        groups = {}
        for cid in sorted(c):
            if c[cid]["qualified"] and c[cid][f] and (allowed is None or cid in allowed):
                groups.setdefault(c[cid][f], []).append(cid)
        if len(groups) < 2:
            return None
        order, lines, nums = [], [f"[비교] {FEAT_NAME[f]}별 사례의 댓글 반응 (댓글 표본이 충분한 사례만, 최상위 댓글 기준)"], {}
        for val, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            start = len(order) + 1
            order += ids
            nums[val] = list(range(start, len(order) + 1))
            if len(ids) == 1:   # 사례 1개: 그 사례 수치를 그대로
                lines.append(f"- {val}: 사례 1개 — {_row(ids[0], start)}")
            else:               # 여러 개: 범위만 한 줄, 사례별 수치는 UI 표
                negs = [c[i]["stats"]["neg_pct"] for i in ids]
                lines.append(f"- {val}: 사례 {len(ids)}개 [{start}]~[{len(order)}] — 부정 {min(negs)}~{max(negs)}% (사례별 수치는 표)")
        few = [v for v, ids in groups.items() if len(ids) < 2]
        if few:
            lines.append(f"[주의] {', '.join(few)} 값은 사례가 1개뿐이라 그 사례 하나의 관찰 결과입니다. "
                         f"{FEAT_NAME[f]} 때문에 반응이 달라졌다고 일반화할 수 없습니다.")
    rows = [{"번호": f"[{n}]", "사례": names()[i], **{FEAT_NAME[f]: c[i][f] or "-" for f in FEATS},
             "댓글 수": c[i]["stats"]["top_related"], "부정 %": c[i]["stats"]["neg_pct"], "긍정 %": c[i]["stats"]["pos_pct"],
             "부정 주된 이유": _rname(c[i]["stats"]["neg_main"]), "긍정 주된 이유": _rname(c[i]["stats"]["pos_main"])}
            for n, i in enumerate(order, 1)]
    groups = {v: n for v, n in nums.items() if len(n) >= 2} if plan["mode"] == "feature" else {}
    return {"head": "\n".join(lines), "case_ids": order, "rows": rows,
            "groups": groups}   # 사례가 여러 개인 값 → 그 값 사례 번호 (묶음 요약 문장 검사용)