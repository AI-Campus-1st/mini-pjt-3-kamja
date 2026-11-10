"""
챗봇과 평가 하네스가 함께 부르는 RAG 체인 (가이드: LangChain 기반, 같은 rag/chain.py).
흐름(기획서 2-1 ⑥~⑩): 범위 판정(밖이면 거절) → 의도·특징 추출 → (예상 외) 비교 질문이면 코드 비교 표 → (예상이면) 비슷한 사례·등급 계산 → KBRetriever 검색 → 답변
- retrieve(): 검색 단계까지 (평가 하네스가 쓰는 부분, 출력 모양 그대로)
- prepare(): 검색 + 코드가 쓰는 부분(등급·수치 문장, 근거 사례, 출처 목록, 주의)
- answer(): prepare → LLM → 코드 검사 (LCEL 체인). 본문은 짧게, 근거 사례 표·출처는 데이터로 (UI가 표시)
  · 예상 답변: 사례마다 LLM을 따로 부름(batch). LLM은 댓글 하나 고르기 + 반응 요약 한 구절.
    인용은 자료와 글자 그대로 대조, 구절은 금지 표현 검사. 실패하면 사례 요약의 대표 댓글(코드가 뽑은 것)로 채움
  · 그 밖의 질문: LLM 한 번, 문장마다 근거 번호
- stream_body(): UI 스트리밍용
확인 실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m rag.chain "질문" [--full: 근거 사례·출처까지]
"""

import re, sys
from itertools import zip_longest
from functools import lru_cache

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag.features import extract
from rag.compare import build as build_compare, detect as detect_compare, fact_head, named_cases
from rag.prompts import (ACCURACY, CASE_HUMAN, CASE_SYSTEM, CAUTION, DATE_NOTE, FEAT_NAME, HUMAN, MAX_QUOTE,
                         REASON_NAME, SYSTEM, TASK_COMPARE, TASK_OTHER)
from rag.reaction import cases, predict
from rag.retriever import QUALIFIED, KBRetriever, get_summaries, to_dict
from rag.scope import check_scope
from rag.rewrite import rewrite
from rag.filters import allowed_cases, label as filter_label, where as filter_where

SCHEME = "time"     # exp2_chunk 결정 (Hit@3 동률, MRR 0.609 > 0.587)
ANSWER_K = 5        # 답변에 넣는 검색 청크 수 (Day 4 결정, Day 5 exp4_topk 후보)
SCORE_MIN = None    # 예상 외 질문의 검색 1위 유사도가 이보다 낮으면 "확인할 수 없음" (None = 끔, 값은 분포를 본 뒤 결정)
NOT_FOUND = "제공된 자료에서 확인할 수 없습니다."
LLM = "gpt-4.1-mini"
MAX_TOKENS = 400    # 설명이 길어지는 것을 막는 상한 (형식 지시의 보조)
NUM = re.compile(r"\d+(\.\d+)?\s*(%|개|명|배|건)")
FORECAST = re.compile(r"(일|할|될|올|볼|보일|날|줄)\s*(것|가능성|수\s*있)")   # "부정적일 것", "나올 수 있" 같은 예상 표현
BAN_WORDS = ["관찰됨", "근거 부족", "가능성", "예상", "전망", "대부분", "압도적", "과반", "명사구", "20자"]   # 마지막 둘: 지시문 베끼기
MAX_PHRASE = 25
TYPE_TAG = re.compile(r"\s*[\[(](소비자 반응|언론 보도|정부 기준)[\])]")   # LLM이 직접 쓴 종류 표시 (지움)
GENERAL = ["영향을 미", "영향을 준", "영향을 주", "차이가 있", "차이를 보", "전반적", "일반적", "보장", "경향", "상대적", "대부분", "대체로"]   # 비교 설명의 일반화 표현
SENT = re.compile(r"(?<=[.!?])\s+")
SENT_SEP = re.compile(r"(?<=[.!?])(\s+)")   # 스트리밍: 문장 사이 공백·줄바꿈을 살려 보여 주기
CONNECT = re.compile(r"^(반면|하지만|그러나|또한|한편|예를 들어)[,\s]*")
QUOTE_LINE = re.compile(r'^\s*-\s*"(.+)"\s*$')   # 사례 요약의 대표 댓글 줄
DIR_KEY = {"부정": ("neg_main", "부정 주된 이유"), "긍정": ("pos_main", "긍정 주된 이유")}
REACT_WORDS = ["반응", "반발", "소비자", "댓글"]   # exp10: 종합 질문 = 반응 단어 + 기준 단어 (코드 판정, LLM 호출 없음)
RULE_WORDS = ["저작권", "정부", "기준", "안내서", "유의점", "주의사항", "쟁점"]
CASE_SIDE = {"doc_type": {"$nin": ["gov", "news_provided"]}}   # 사례 요약·댓글
GOV_SIDE = {"doc_type": "gov"}


def is_synthesis(q):
    return any(w in q for w in REACT_WORDS) and any(w in q for w in RULE_WORDS)


def _and(a, b):
    return {"$and": [a, b]} if a and b else (a or b)


def retrieve(q, k=3, filters=None, info=None):
    """filters: rag.filters.normalize 결과 (None이면 지금과 똑같이 검색 → 평가 하네스·기준선 그대로)
    info: 이미 뽑은 {intent, features} (기획서 확인 탭이 표와 같은 특징으로 답하도록). None이면 지금처럼 질문에서 뽑음"""
    info = info or extract(q)
    pred, where, allowed = None, None, allowed_cases(filters)
    if info["intent"] == "predict":
        pred = predict(info["features"], allowed=allowed)
        ids = [s["case_id"] for s in pred["similar"]]
        where = {"case_id": {"$in": ids}} if ids else QUALIFIED   # 비슷한 사례가 없으면 탈락 사례만 빼고 검색
    fw = filter_where(filters)
    synth = pred is None and is_synthesis(q)
    if synth:   # exp10_source_quota: 반응(사례·댓글)과 기준(정부)을 나눠 찾아 번갈아 섞음 (반응 먼저)
        half = k // 2
        a = KBRetriever(k=k - half, scheme=SCHEME, where=_and(CASE_SIDE, fw)).invoke(q)
        b = KBRetriever(k=half, scheme=SCHEME, where=_and(GOV_SIDE, fw)).invoke(q)
        docs = [d for pair in zip_longest(a, b) for d in pair if d is not None]
    else:
        where = {"$and": [where, fw]} if where and fw else (where or fw)
        docs = KBRetriever(k=k, scheme=SCHEME, where=where).invoke(q)
    return {**info, "prediction": pred, "synthesis": synth, "docs": [to_dict(d) for d in docs]}


# ---------- 코드가 쓰는 부분 (등급·수치는 여기서만 문장이 됨) ----------

def _rname(m):
    return REASON_NAME[m["code"]] if m else "-"


def _case_rows(pred, nums, titles):
    """UI 근거 사례 표용 데이터 (본문에는 넣지 않음)"""
    rows = []
    for i, s in zip(nums, pred["similar"]):
        c = cases()[s["case_id"]]
        rows.append({"번호": f"[{i}]", "사례": titles.get(s["case_id"], s["case_id"]),
                     **{FEAT_NAME[f]: c.get(f) or "-" for f in FEAT_NAME}, "일치 특징": s["n_match"],
                     "댓글 수": s["top_related"], "부정 %": s["neg_pct"], "긍정 %": s["pos_pct"],
                     "부정 주된 이유": _rname(s["neg_main"]), "긍정 주된 이유": _rname(s["pos_main"])})
    return rows


def _quote_plan(pred, nums):
    """사례마다 인용할 방향 = 비율이 더 큰 쪽(같으면 부정), 그 방향 주된 이유가 없으면 다른 쪽. 둘 다 없으면 인용 안 함"""
    plan = {}
    for i, s in zip(nums, pred["similar"]):
        order = ["부정", "긍정"] if s["neg_pct"] >= s["pos_pct"] else ["긍정", "부정"]
        for d in order:
            m = s[DIR_KEY[d][0]]
            if m:
                plan[i] = {"dir": d, "reason": REASON_NAME[m["code"]]}
                break
    return plan


def _reason_line(pred, d, label):
    cnt = pred[f"{d}_reasons"]
    if not cnt:
        return f"- {label} 반응의 주된 이유: 정할 만큼 {label} 댓글이 있는 사례가 없음"
    top = max(cnt.values())
    names = "', '".join(REASON_NAME[c] for c, v in cnt.items() if v == top)
    return f"- {label} 반응의 주된 이유: '{names}' (사례 {len(pred['similar'])}개 중 {top}개){' (동률)' if pred[f'{d}_tie'] else ''}"


def _head(info, pred, nums):
    n = len(pred["similar"])
    if pred["grade"] == "관찰됨":
        return "\n".join([f"[예상] 관찰됨 — 비슷한 사례 {n}개(특징 2개 이상 일치)에서 아래 반응이 나온 사례가 있습니다. "
                          f"{''.join(f'[{x}]' for x in nums)}", _reason_line(pred, "neg", "부정"), _reason_line(pred, "pos", "긍정")])
    if n == 1:
        return "[예상] 근거 부족 — 비슷한 사례가 1개뿐이라 예상하기 어렵습니다. 아래는 그 사례 하나의 관찰 결과입니다. [1]"
    if not any(info["features"].values()):
        return "[예상] 근거 부족 — 질문에서 광고 특징(업종·AI를 쓴 부분·공개 여부·시기)을 찾지 못해 비슷한 사례를 고를 수 없습니다."
    return "[예상] 근거 부족 — 특징이 2개 이상 같은 사례가 없어 예상하기 어렵습니다."


def _group(docs):
    """같은 문서(영상·기사)의 청크는 출처 번호 하나로 합침. 정부 문서는 쪽이 근거라 청크마다 따로"""
    groups = {}
    for d in docs:
        m = d["meta"]
        key = d["id"] if m.get("doc_type") == "gov" else m.get("doc_id", d["id"])
        groups.setdefault(key, []).append(d)
    return list(groups.values())


def _source(i, chunks):
    m = chunks[0]["meta"]
    ps, pe = m.get("page_start"), m.get("page_end")
    pages = "" if ps is None else f"{ps}쪽" if ps == pe else f"{ps}~{pe}쪽"
    return {"번호": i, "case_id": m.get("case_id"), "종류": m.get("authority", ""), "제목": m.get("display_name") or m.get("title", ""), "원제목": m.get("title", ""), "발행처": m.get("source", ""),
            "발행일": m.get("published_at", ""), "쪽": pages, "링크": m.get("url", ""),
            "청크": [{"id": c["id"], "text": c["text"], "score": c["score"]} for c in chunks]}   # 원문 펼쳐 보기·유사도용


def _ctx(src):
    head = f"[{src['번호']}] ({src['종류']}) {src.get('원제목', src['제목'])} | {src['발행처']} | 발행일 {src['발행일']} {src['쪽']}".rstrip()
    return head + "\n" + "\n---\n".join(c["text"] for c in src["청크"])


def _summary_quotes(text):
    """사례 요약의 대표 댓글 (코드가 주된 이유별 좋아요 순으로 뽑은 것) → {"부정": [...], "긍정": [...]}"""
    out, sec = {"부정": [], "긍정": []}, None
    for line in text.splitlines():
        if line.startswith(("부정 댓글의 주된 이유", "긍정 댓글의 주된 이유")):
            sec = line[:2]
        elif sec and QUOTE_LINE.match(line):
            out[sec].append(QUOTE_LINE.match(line).group(1))
        else:
            sec = None
    return out


def _refusal(q, head, why, r=None):
    """거절 답변 (LLM·출처 없음). why: 범위 밖 유형 또는 low_score"""
    return {"question": q, "task": "", "context": "", "case_inputs": [], "is_predict": False, "refusal": why,
            "is_compare": False, "compare": None, "n_group_table": 0, "groups": {},
            "notice": "", "scope": why,
            "head": head, "foot": "", "skip_llm": True, "cases": [], "excluded": [], "accuracy": "",
            "sources": [], "retrieval": r}


def prepare(q, progress=None, history=None, filters=None, info=None):
    """멀티턴: history가 있으면 질문을 먼저 다시 쓴 뒤 그 질문으로 처리 (없으면 그대로 → 평가·기준선 영향 없음)
    결과의 question = 처리한 질문, asked = 사용자가 입력한 질문, rewrite = 다시 쓰기 기록"""
    if history:   # 원래 질문이 범위 밖(legal 제외)이면 다시 쓰지 않고 거절 — 다시 쓴 질문에 앞 특징이 섞여 거절을 놓치는 것 방지
        sc0 = check_scope(q)
        if sc0["scope"] and not sc0["soft"]:
            p = _refusal(q, sc0["message"], sc0["scope"])
            p.update({"asked": q, "rewrite": {"question": q, "rewritten": False, "error": None}, "filters": filter_label(filters)})
            return p
        if progress:
            progress("⓪ 앞 대화를 보고 질문 정리 중")
    rw = rewrite(q, history)
    p = _prepare(rw["question"], progress, filters, info)
    p.update({"asked": q, "rewrite": rw, "filters": filter_label(filters)})
    return p


def _prepare(q, progress=None, filters=None, info=None):
    """progress: UI 진행 상태 표시용 함수 (없으면 아무것도 안 함). 결과에는 영향 없음"""
    step = progress or (lambda m: None)
    step("① 범위 판정 중 — 자료로 답할 수 있는 질문인지")
    sc = check_scope(q)                      # ① 범위 밖이면 검색 없이 거절 (기획서 1-3). legal은 약한 거절
    if sc["scope"] and not sc["soft"]:
        return _refusal(q, sc["message"], sc["scope"])
    notice = sc["message"] if sc["soft"] else ""
    step("② 특징 추출 → 사례 검색 → 등급 판정 중")
    r = retrieve(q, k=ANSWER_K, filters=filters, info=info)
    if r["intent"] != "predict" and SCORE_MIN is not None and (not r["docs"] or r["docs"][0]["score"] < SCORE_MIN):
        return _refusal(q, NOT_FOUND, "low_score", r)   # ② 관련 자료가 약하면 거절 (지금은 꺼 둠: 점수 분포가 겹침)
    pred = r["prediction"]
    skip = bool(pred) and not pred["similar"]   # 예상인데 비슷한 사례 0개: 근거가 없으므로 LLM·출처 모두 쓰지 않음
    cmp_plan, comp = None, None
    if not pred:                                 # ③ 비교 질문이면 코드가 비교 표를 만듦 (Day 4 추가)
        step("③ 비교 질문인지 판정 중")
        cmp_plan = detect_compare(q)
        comp = build_compare(cmp_plan, allowed_cases(filters)) if cmp_plan["mode"] else None
    fact = None
    if not pred and not comp:                    # ④ exp13_fact_head: 질문에 사례 이름이 하나면 코드가 그 사례 수치 줄을 붙임
        named, allowed = named_cases(q), allowed_cases(filters)
        fact = named[0] if len(named) == 1 and (allowed is None or named[0] in allowed) else None
    sums = get_summaries([s["case_id"] for s in pred["similar"]] if pred else comp["case_ids"] if comp else [fact] if fact else [])
    titles = {d["meta"]["case_id"]: d["meta"].get("title", d["meta"]["case_id"]) for d in sums}
    names = {d["meta"]["case_id"]: d["meta"].get("display_name") or titles[d["meta"]["case_id"]] for d in sums}   # 화면 표시용
    seen = {d["id"] for d in sums}
    groups = [] if skip else [[d] for d in sums] + _group([d for d in r["docs"] if d["id"] not in seen])
    sources = [_source(i, g) for i, g in enumerate(groups, 1)]
    nums = list(range(1, len(sums) + 1))

    case_inputs = []
    if pred and not skip:
        head, case_rows, foot = _head(r, pred, nums), _case_rows(pred, nums, names), CAUTION
        plan = _quote_plan(pred, nums)
        if pred["grade"] != "관찰됨":
            plan = {k: v for k, v in plan.items() if k == 1}   # 근거 부족: 그 사례 하나의 관찰 결과만
        for i, s, d in zip(nums, pred["similar"], sums):
            if i not in plan:
                continue
            mine = [src for src in sources if src["case_id"] == s["case_id"]]   # 이 사례의 요약 + 검색된 댓글 묶음만
            case_inputs.append({"n": i, "title": titles[s["case_id"]], "name": names[s["case_id"]], **plan[i], "sources": mine,
                                "context": "\n\n".join(_ctx(src) for src in mine),
                                "fallback": _summary_quotes(d["text"])[plan[i]["dir"]]})
    elif pred:
        head, case_rows, foot = _head(r, pred, nums), [], ""
    elif comp:
        head, case_rows, foot = comp["head"], comp["rows"], CAUTION
    elif fact:
        head, case_rows, foot = fact_head(fact), [], CAUTION
    else:
        head, case_rows, foot, titles = "", [], "", {}
    ctx = "\n\n".join(_ctx(s) for s in sources)
    return {"question": q, "task": TASK_COMPARE if comp else TASK_OTHER, "context": f"{head}\n\n{ctx}" if comp or fact else ctx, "fact_case": fact,
            "is_compare": bool(comp), "compare": cmp_plan,
            "n_group_table": len(comp["case_ids"]) if comp and cmp_plan["mode"] == "feature" else 0,
            "groups": comp["groups"] if comp else {},
            "case_inputs": case_inputs, "is_predict": bool(pred), "refusal": None,
            "notice": notice, "scope": sc["scope"],
            "head": head, "foot": foot, "skip_llm": skip or (bool(pred) and not case_inputs), "cases": case_rows,
            "excluded": pred["excluded"] if pred else [],
            "excluded_names": [d["meta"].get("display_name") or d["meta"].get("title", d["meta"]["case_id"])
                               for d in get_summaries(pred["excluded"])] if pred and pred["excluded"] else [], "accuracy": ACCURACY if case_rows else "",
            "sources": sources, "retrieval": r}


def _norm(t):
    return " ".join(str(t).split())


def _short(q):
    return q if len(q) <= MAX_QUOTE else q[:MAX_QUOTE] + "…"


def phrase_problem(ph):
    """반응 구절 검사. 문제 없으면 None"""
    if not ph or len(ph) > MAX_PHRASE:
        return "구절 길이"
    if NUM.search(ph):
        return "구절 수치"
    if FORECAST.search(ph) or any(w in ph for w in BAN_WORDS):
        return "구절 예상·지시문 표현"
    return None


def render_case(ci, res):
    """사례 하나의 LLM 결과 → (코드가 쓴 줄, 기록). 인용 실패 → 요약 대표 댓글, 구절 실패 → 구절 없이"""
    log = {"case": ci["n"], "llm": res if isinstance(res, dict) else repr(res), "quote_src": "llm", "problems": []}
    res = res if isinstance(res, dict) else {}
    quote, ph = _norm(res.get("quote", "")), _norm(res.get("phrase", ""))
    ref = None
    if quote:
        ref = next((src["번호"] for src in ci["sources"] if quote in _norm(_ctx(src))), None)
    if not ref:
        log["problems"].append("자료에 없는 인용" if quote else "인용 없음")
        log["quote_src"] = "summary" if ci["fallback"] else "none"
        quote, ref, ph = (ci["fallback"][0], ci["n"], "") if ci["fallback"] else ("", None, "")
    elif (why := phrase_problem(ph)):
        log["problems"].append(why)
        ph = ""
    name = ci.get("name", ci["title"])
    log.update({"quote": quote, "ref": ref, "title": name, "dir": ci["dir"]})   # UI: 인용 댓글 (대표 고르기·원문 펼침)
    label = f"- [{ci['n']}] {name} — {ci['dir']} · {ci['reason']}"
    if not quote:
        return label + ": 인용할 대표 댓글 없음", log
    return f"{label}: \"{_short(quote)}\"{' → ' + ph if ph else ''} [{ref}]", log


# ---------- LangChain 체인 ----------

def _llm(**kw):
    return ChatOpenAI(model=LLM, temperature=0, max_tokens=MAX_TOKENS,
                      default_headers={"Accept-Encoding": "gzip, deflate"}, **kw)   # 키는 .env


@lru_cache(maxsize=1)
def _gen():
    """예상 외 질문: 한 번 부름"""
    return ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)]) | _llm() | StrOutputParser()


@lru_cache(maxsize=1)
def _case_gen():
    """예상 질문: 사례마다 따로 부름 (JSON 출력)"""
    llm = _llm().bind(response_format={"type": "json_object"})
    return ChatPromptTemplate.from_messages([("system", CASE_SYSTEM), ("human", CASE_HUMAN)]) | llm | JsonOutputParser()


def _case_lines(p):
    """사례별 호출을 동시에 보내고(batch), 결과를 사례 순서대로 코드가 검사·조립"""
    res = _case_gen().batch(p["case_inputs"], config={"max_concurrency": 5}, return_exceptions=True)
    return [render_case(ci, r) for ci, r in zip(p["case_inputs"], res)]


def tag_sources(body, sources):
    """예상 외 답변: LLM이 직접 쓴 종류 표시는 지움(틀리게 붙이는 일 방지, 종류는 출처 목록에서).
    정부 자료를 인용했으면 발행 시기 안내 한 줄을 붙임"""
    body = TYPE_TAG.sub("", body).strip()
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    gov = any(sources[n - 1]["종류"] == "정부 기준" for n in cited if 1 <= n <= len(sources))
    return f"{body}\n\n{DATE_NOTE}" if gov else body


def _group_talk(sent, cites, groups):
    """사례가 여러 개인 값(예: 영상 전체)을 말하면서 그 값의 사례 하나만 인용한 게 아니면 True (묶음 요약)"""
    for val, nums in groups.items():
        if val in sent and not (len(cites & set(nums)) == 1 and len(cites & set(range(1, max(nums) + 1))) <= 1):
            return True
    return False


def drop_general(body, n_table=0, groups=None):
    """비교 설명: 수치·일반화 표현이 든 문장은 지움 (수치는 코드 표에 있음). 지운 문장은 기록
    n_table > 0 (특징 값 비교): 표의 사례 번호를 2개 이상 인용하거나, 사례가 여러 개인 값을 묶어 말한 문장도 지움
    → 여러 사례 묶음 요약은 코드 표만 함"""
    kept, dropped = [], []
    for sent in filter(str.strip, SENT.split(body.strip())):
        why = drop_why(sent, n_table, groups)
        (dropped.append({"line": sent, "why": why}) if why else kept.append(sent))
    if kept:   # 앞 문장이 지워져 접속어로 시작하면 뗌
        kept[0] = CONNECT.sub("", kept[0])
    return " ".join(kept), dropped


def drop_why(sent, n_table=0, groups=None):
    """비교 설명 한 문장을 지울 이유 (없으면 None). drop_general과 스트리밍이 같이 씀"""
    cites = {int(n) for n in re.findall(r"\[(\d+)\]", sent)}
    n_cases = len({n for n in cites if 1 <= n <= n_table})
    return ("근거 번호 없음" if not cites else "수치" if NUM.search(sent) else "일반화" if any(w in sent for w in GENERAL)
            else "여러 사례 묶음" if n_cases >= 2 or _group_talk(sent, cites, groups or {}) else None)


def _body(p):
    if p["skip_llm"]:
        return {"body": "", "case_log": [], "cite_problems": [], "dropped": []}
    if p["is_predict"]:
        lines = _case_lines(p)
        return {"body": "\n".join(l for l, _ in lines), "case_log": [g for _, g in lines], "cite_problems": [], "dropped": []}
    return _post(p, _gen().invoke(p))


def _post(p, raw):
    """예상 외 질문: LLM 원문 → 검사 끝난 본문 (answer와 stream_body가 같은 함수를 씀)"""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", raw)}
    bare = re.sub(r"\[\d+\]", "", raw).strip()
    if NOT_FOUND.rstrip(".") in bare and len(SENT.split(bare)) == 1:   # 자료로 답할 수 없음: 번호·출처를 붙이지 않음
        return {"body": NOT_FOUND, "raw_body": raw, "case_log": [], "dropped": [], "cite_problems": [], "sources": []}
    body, dropped = drop_general(raw, p["n_group_table"], p["groups"]) if p["is_compare"] else (raw, [])
    return {"body": tag_sources(body, p["sources"]) if body else "", "raw_body": raw, "case_log": [], "dropped": dropped,
            "cite_problems": sorted(n for n in cited if not 1 <= n <= len(p["sources"]))}


def _join(p):
    """본문 = 등급 몇 줄 + 사례별 한 줄(또는 설명) + 주의 한 줄. 근거 사례 표·출처 목록은 UI가 cases·sources로 그림"""
    return "\n\n".join(x for x in [p["notice"], p["head"], p["body"], p["foot"]] if x)


@lru_cache(maxsize=1)
def rag_chain():
    return RunnableLambda(prepare) | RunnableLambda(lambda p: {**p, **_body(p)}) \
        | RunnablePassthrough.assign(text=RunnableLambda(_join))


def answer(q):
    """질문 → {text, head, body, case_log(사례별 LLM 원문·대체 여부·문제), foot, cases(표), sources, retrieval, ...}"""
    return rag_chain().invoke(q)


def stream_body(p, out=None):
    """UI 스트리밍: 검사를 통과한 것만 순서대로 내보냄. 끝나면 out에 answer()의 _body와 같은 모양의 결과를 채움
    - 예상: 사례별 호출이 끝나는 대로 받되, 사례 번호 순서대로 한 줄씩
    - 그 밖: LLM 글을 받으며 문장이 끝날 때마다 검사(종류 표시 지우기, 비교면 drop_why)한 뒤 내보냄.
      "확인할 수 없습니다"가 든 첫 문장은 뒤에 문장이 더 오는지 볼 때까지 붙잡아 둠 (한 문장뿐이면 번호·출처를 지우므로)
    - 최종 본문은 마지막에 _post(원문 전체)로 한 번 더 만듦 → 화면은 out["body"]로 덮어써 answer()와 같게"""
    out = {} if out is None else out
    if p["skip_llm"]:
        out.update(_body(p))
        return
    if p["is_predict"]:
        cis, got, nxt = p["case_inputs"], {}, 0
        for i, r in _case_gen().batch_as_completed(cis, config={"max_concurrency": 5}, return_exceptions=True):
            got[i] = render_case(cis[i], r)
            while nxt in got:
                yield got[nxt][0] + "\n"
                nxt += 1
        lines = [got[i] for i in range(len(cis))]
        out.update({"body": "\n".join(l for l, _ in lines), "case_log": [g for _, g in lines],
                    "cite_problems": [], "dropped": []})
        return
    raw, buf, held, kept, stem = "", "", None, 0, NOT_FOUND.rstrip(".")

    def show(sent, sep):
        nonlocal kept
        if not sent.strip():
            return ""
        if p["is_compare"]:
            if drop_why(sent, p["n_group_table"], p["groups"]):
                return ""
            sent, sep = (CONNECT.sub("", sent) if not kept else sent), " "
        kept += 1
        return TYPE_TAG.sub("", sent).strip() + sep

    for tok in _gen().stream(p):
        raw += tok
        buf += tok
        parts = SENT_SEP.split(buf)          # [문장, 공백, 문장, 공백, ..., 아직 안 끝난 글]
        buf = parts.pop()
        for sent, sep in zip(parts[::2], parts[1::2]):
            if held is None and not kept and stem in sent:
                held = (sent, sep)           # 첫 문장이 "확인할 수 없습니다": 다음 문장이 올 때까지 대기
                continue
            if held:
                yield show(*held)
                held = None
            yield show(sent, sep)
    final = _post(p, raw)
    out.update(final)
    if final["body"] == NOT_FOUND:
        yield NOT_FOUND
        return
    for x in ([held] if held else []) + [(buf, "")]:
        yield show(*x)
    if final["body"].endswith(DATE_NOTE):
        yield "\n\n" + DATE_NOTE


def _print_full(out):
    """터미널 확인용: UI가 표로 그릴 근거 사례·출처를 글로 출력"""
    for c in out["cases"]:
        print(" ", " | ".join(f"{k} {v}" for k, v in c.items()))
    if out["excluded"]:
        print("  근거에서 뺀 사례(표본 부족):", ", ".join(out["excluded"]))
    for s in out["sources"]:
        sc = ", ".join(str(c["score"]) for c in s["청크"] if c["score"] is not None)
        print(f"[{s['번호']}] {s['종류']} · {s['제목']} · {s['발행처']} · {s['발행일']} {s['쪽']} · {s['링크']}"
              f" (청크 {len(s['청크'])}개{', 유사도 ' + sc if sc else ''})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--full"]
    out = answer(args[0] if args else "식품 광고 영상 전체를 AI로 만들고 AI 사용을 밝히면 반응은?")
    print(out["text"])
    if out["refusal"] or out["notice"]:
        print("\n[거절 유형]", out["refusal"] or f"{out['scope']} (약한 거절: 답변 계속)")
    if out.get("dropped"):
        print("\n[지운 문장]", *[f"\n  ({d['why']}) {d['line']}" for d in out["dropped"]])
    if out["case_log"]:
        print("\n[규칙 확인] 사례별 인용 출처·문제")
        for g in out["case_log"]:
            print(f"  [{g['case']}] 인용 {g['quote_src']} {g['problems'] or ''} | LLM 원문 {g['llm']}")
    if "--full" in sys.argv:
        print("\n[근거 사례·출처: UI에서는 표와 펼침 영역]")
        _print_full(out)