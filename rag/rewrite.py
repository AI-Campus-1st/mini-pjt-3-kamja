"""멀티턴: 이어지는 질문을 혼자서도 이해되는 질문으로 다시 쓰기 (Day 4 오후 4번, 가이드 심화 '질의 재작성')
- 이전 대화가 없으면 LLM을 부르지 않고 질문 그대로 → 평가 하네스(answer(q))와 run1 기준선은 그대로
- 다시 쓴 질문을 chain.prepare가 범위 판정·특징 추출·검색·답변에 그대로 씀. 화면에는 '이렇게 이해했어요'로 보여 줌
- 실패하면 원래 질문을 쓰고 오류를 기록 (삼키지 않음)"""
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL, MAX_TURNS = "gpt-4.1-mini", 3

SYSTEM = """너는 대화의 마지막 질문을 앞 대화 없이도 이해되는 질문 하나로 다시 쓴다. JSON 객체 하나만 출력한다.
- 마지막 질문이 앞 대화를 가리키면("그럼", "그건", "거기서", "비교하면", 생략된 대상) 앞 질문에서 필요한 대상만 채운다:
  광고 특징(업종, AI를 쓴 부분, AI 사용 공개 여부), 사례 이름, 주제(소비자 반응·저작권 기준 등)
- 마지막 질문이 바꾼 부분은 마지막 질문을 따른다 (예: 업종만 바꾸면 업종만 바꾸고 나머지 특징은 유지)
- 앞 대화에 없는 내용·수치·판단을 더하지 않는다. 앞 답변의 결론을 질문에 넣지 않는다
- 마지막 질문이 이미 혼자서 이해되면 한 글자도 바꾸지 않고 그대로 둔다. 묻는 대상(수치·비용·법률 등)을 앞 질문 특징과 섞지 않는다
- 특징 하나를 바꾸면 앞 질문의 그 특징 단어를 지우고 새 값으로 바꿔 쓴다. 두 값을 함께 남기지 않는다
예시 (값은 예시일 뿐)
- 앞: 게임 광고 배경만 AI로 만들면 반응은? / 마지막: 공공기관이면? → 공공기관 광고 배경만 AI로 만들면 반응은?
- 앞: 게임 광고 배경만 AI로 만들면 반응은? / 마지막: AI 광고 제작 비용은 보통 얼마야? → AI 광고 제작 비용은 보통 얼마야?
출력 형식: {{"standalone": "다시 쓴 질문"}}"""


# 업종 단어 (features.py ALLOWED["industry"]를 사람이 쓰는 말로 풀어 둠)
INDUSTRY = ["뷰티", "화장품", "식품", "유통", "통신", "IT", "금융", "보험", "자동차", "교육", "공공기관", "방송", "게임"]


def fix_industry(follow, out):
    """코드 보정: 이어지는 질문이 업종을 새로 말했으면, 다시 쓴 질문에 남은 앞 업종 단어를 지움 (LLM이 두 업종을 함께 남기는 것 방지)"""
    new = [w for w in INDUSTRY if w in follow]
    if not new:
        return out
    for w in INDUSTRY:
        if w not in follow and w in out:
            out = re.sub(rf"(?<![가-힣A-Za-z]){re.escape(w)}\s*(?=(광고|업종|분야))", "", out, count=1)   # 'OO 광고·업종'일 때만 (사례명 속 단어는 그대로)
    return " ".join(out.split())


@lru_cache(maxsize=1)
def _chain():
    llm = ChatOpenAI(model=MODEL, temperature=0, default_headers={"Accept-Encoding": "gzip, deflate"}) \
        .bind(response_format={"type": "json_object"})
    return ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{dialog}")]) | llm | JsonOutputParser()


def rewrite(q, history=None):
    """history: [{"q": 이전 질문(다시 쓴 것), "a": 이전 답변 요약}] → {"question", "rewritten", "error"}"""
    if not history:
        return {"question": q, "rewritten": False, "error": None}
    turns = "\n".join(f"사용자: {h['q']}\n챗봇: {h['a']}" for h in history[-MAX_TURNS:])
    try:
        raw = (_chain().invoke({"dialog": f"[앞 대화]\n{turns}\n\n[마지막 질문]\n{q}"}).get("standalone") or "").strip()
        out = fix_industry(q, raw) if raw else ""
        return {"question": out or q, "rewritten": bool(out) and out != q, "error": None,
                "llm": raw, "fixed": out != raw}   # 코드 보정 여부 기록
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:300]
        print(f"[rewrite 오류] {err}")
        return {"question": q, "rewritten": False, "error": err}


def summary(r):
    """이전 답변을 다시 쓰기용으로 짧게 (머리말 첫 줄 + 본문 앞부분)"""
    head = (r.get("head") or "").split("\n")[0]
    return f"{head} {(r.get('body') or '')[:200]}".strip() or "(답변 없음)"