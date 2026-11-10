"""
범위 밖 질문 판정 (기획서 1-3 답변 불가 유형, 3-4 수집 제외, 3-6 커버리지 한계).
특징 추출(features.py)과 따로 부름 → 의도·특징·등급 기준선(run1)에 영향 없음. 거절 문구는 코드가 정함
"""

from functools import lru_cache

from dotenv import load_dotenv

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()   # 이 모듈만 불러와도 .env의 키를 읽도록 (평가 스크립트에서 단독 사용)
MODEL = "gpt-4.1-mini"

# 유형: (판정 기준, 거절 문구). 문구 끝에 할 수 있는 일을 함께 안내
SCOPE = {
    "numbers": ("매출·비용·광고 효과·구매 전환 같은 수치나 성과를 묻는 질문 (기획서 1-3)",
                "매출·비용·광고 효과 같은 수치 자료는 이 챗봇의 자료에 없어 답할 수 없습니다. "
                "대신 비슷한 국내 사례의 유튜브 댓글 반응은 보여 드릴 수 있어요."),
    "legal": ("특정 캠페인·이미지·상황이 합법인지, 소송·처벌을 받는지 같은 개별 법률 판단 (기획서 1-3)",
              "[안내] 개별 사안의 합법·위법은 판단하지 않습니다(필요하면 법률 전문가와 상담하세요). "
              "아래는 자료에 있는 일반 기준과 사례 반응입니다."),
    "labeling": ("AI 생성물 표시 의무·표시 방법·워터마크 규정 (기획서 1-3, 3-6)",
                 "AI 생성물 표시 의무 규정은 이번 자료에 넣지 않아 답할 수 없습니다. "
                 "AI 사용을 밝힌 사례와 밝히지 않은 사례의 댓글 반응은 보여 드릴 수 있어요."),
    "foreign": ("해외 기업·해외 캠페인 사례의 반응 (기획서 3-4, 3-6)",
                "국내 사례만 다루고 있어 해외 캠페인 반응은 답할 수 없습니다. 비슷한 국내 사례는 찾아 드릴 수 있어요."),
    "platform": ("유튜브가 아닌 플랫폼(인스타그램·X·틱톡·블로그·커뮤니티 등)에서의 반응 (기획서 3-4, 3-6)",
                 "반응 자료는 유튜브 댓글만 모았기 때문에 다른 플랫폼의 반응은 답할 수 없습니다. 같은 사례의 유튜브 댓글 반응은 보여 드릴 수 있어요."),
    "audience": ("연령·성별·지역 등 댓글 작성자 특성별 반응 차이 (기획서 3-6: 작성자 정보 미저장)",
                 "개인정보 보호를 위해 댓글 작성자 정보를 저장하지 않아 연령·성별별 반응은 나눠 볼 수 없습니다. 사례 전체의 댓글 반응은 보여 드릴 수 있어요."),
    "future": ("앞으로의 규제·법 개정·시장 변화에 대한 전망 (기획서 3-6: 자료는 발행일 기준)",
               "앞으로의 규제·시장 변화를 전망할 자료는 없습니다. 현재 정부 안내서의 기준(발행일 표시)은 안내해 드릴 수 있어요."),
}

# 약한 거절: 거절하지 않고 판단 거절 문구만 앞에 붙인 뒤 원래 경로로 답함 (Day 4 결정)
# "괜찮을까?"·"문제가 되나?" 같은 일상 표현이 법률 질문으로 판정돼 정상 질문 5개를 거절했기 때문 (day4_scope_v2)
SOFT = {"legal"}

PROMPT = """너는 AI 이미지 광고 반응 안내 챗봇의 질문 범위를 판정한다. JSON 객체 하나만 출력한다.
이 챗봇이 답하는 것: 국내 AI 광고 사례 11개의 유튜브 댓글 반응, 그 반응을 근거로 한 새 기획의 반응 예상,
정부 안내서의 AI 저작권 일반 기준(등록 가능 여부 등), 제공된 뉴스의 내용.
아래 유형 중 하나에 해당하면 그 이름을, 아니면 "none"을 고른다.
""" + "\n".join(f"- {k}: {v[0].split(' (기획서')[0]}" for k, v in SCOPE.items()) + """
주의
- 앞으로 만들 광고의 소비자 반응(댓글 반응)을 묻는 예상 질문은 none이다. 매출·비용 같은 수치를 물을 때만 numbers
- 정부 기준이 일반적으로 어떤지 묻는 질문은 none, 우리 캠페인·특정 이미지가 괜찮은지 판단을 구하면 legal
- 확실하지 않으면 none
출력 형식: {{"scope": "..."}}"""


@lru_cache(maxsize=1)
def _chain():
    llm = ChatOpenAI(model=MODEL, temperature=0, default_headers={"Accept-Encoding": "gzip, deflate"}) \
        .bind(response_format={"type": "json_object"})   # 키는 .env
    return ChatPromptTemplate.from_messages([("system", PROMPT), ("human", "{q}")]) | llm | JsonOutputParser()


def check_scope(q):
    """질문 → {"scope": 유형 또는 None, "soft": 약한 거절 여부, "message": 문구, "error": 오류 내용 또는 None}
    허용값 밖·오류는 범위 안(None)으로 보고 답변을 이어감. 오류는 삼키지 않고 기록"""
    try:
        s, err = _chain().invoke({"q": q}).get("scope"), None
    except Exception as e:
        s, err = None, f"{type(e).__name__}: {e}"[:300]
        print(f"[scope 오류] {err}")
    s = s if s in SCOPE else None
    return {"scope": s, "soft": s in SOFT, "message": SCOPE[s][1] if s else "", "error": err}