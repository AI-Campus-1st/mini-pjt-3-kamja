"""
질문에서 의도(예상 질문인지)와 사례 특징 4개를 뽑는다 (기획서 2-1 ⑥). 질문 분석만 하고 문서는 임베딩하지 않는다
- 허용값 밖의 답은 코드에서 None으로 바꾼다 (빈 특징값은 어떤 사례와도 일치하지 않음, 2-5)
- exp9_feat_prompt: ai_part 값마다 뜻풀이 한 줄 추가 (바꾼 변수는 이것 하나)
"""

import json, re
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODEL = "gpt-4.1-mini"
ALLOWED = {"industry": ["뷰티", "식품", "유통", "통신·IT", "금융·보험", "자동차", "교육", "공공기관", "기타(방송)", "기타(게임)", "기타"],
           "ai_part": ["모델 인물", "제품 이미지", "배경·연출 일부", "영상 전체"],
           "disclosed": ["처음부터 공개", "논란 후 인정", "영상 내 표기 없음"]}
PERIOD = re.compile(r"^20\d\d (상반기|하반기)$")

PROMPT = f"""너는 광고 기획 질문을 분석한다. JSON 객체 하나만 출력한다.
- intent: 앞으로 만들 광고·기획에 AI를 쓰면 소비자 반응이 어떨지 묻는 질문이면 "predict",
  이미 있었던 특정 사례의 반응·사례 비교·정부 기준·저작권·그 밖의 질문이면 "other"
- industry: {ALLOWED["industry"]} 중 하나. 화장품·코스메틱은 뷰티, 방송은 기타(방송), 게임은 기타(게임)
- ai_part: {ALLOWED["ai_part"]} 중 하나. AI로 만든 대상으로 고른다.
  모델 인물 = 광고에 나오는 사람(모델·출연자)을 AI로 만듦. 사람이 제품과 함께 나와도 그 사람이 AI면 모델 인물,
  제품 이미지 = 사람 없이 제품 사진·그림만 AI로 만듦,
  배경·연출 일부 = 실제 촬영한 광고에 일부 장면·배경·효과만 AI로 넣음,
  영상 전체 = 광고 영상을 AI로 만듦. "AI로 만든 영상"처럼 일부라고 좁히지 않고 영상을 AI로 만들었다고 하면 영상 전체
- disclosed: {ALLOWED["disclosed"]} 중 하나. AI 사용을 처음부터 밝히면 처음부터 공개,
  AI 표기 없이·밝히지 않고 내보내면 영상 내 표기 없음, 논란 뒤에 인정하면 논란 후 인정
- period: 질문에 시기가 적혀 있으면 "2026 하반기" 형식, 없으면 null
- 질문에 근거가 없는 특징은 추측하지 말고 null
출력 형식: {{"intent": "...", "industry": ..., "ai_part": ..., "disclosed": ..., "period": ...}}"""


@lru_cache(maxsize=1)
def _client():
    return OpenAI(default_headers={"Accept-Encoding": "gzip, deflate"})   # 키는 .env의 OPENAI_API_KEY


def extract(q):
    r = _client().chat.completions.create(model=MODEL, temperature=0, response_format={"type": "json_object"},
                                          messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": q}])
    raw = json.loads(r.choices[0].message.content)
    feats = {k: raw.get(k) if raw.get(k) in v else None for k, v in ALLOWED.items()}
    feats["period"] = raw.get("period") if isinstance(raw.get("period"), str) and PERIOD.match(raw["period"]) else None
    return {"intent": "predict" if raw.get("intent") == "predict" else "other", "features": feats}