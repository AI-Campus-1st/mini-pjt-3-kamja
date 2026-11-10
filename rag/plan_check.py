"""
기획서 확인 (심화, 기획서 9-1): 기획서 → AI 사용 항목·공통 정보(회사·AI 표기 방침·시기) 원문 인용 → 코드가 항목마다 질문 조립(질의 재작성)
→ 항목별 특징(features.extract, exp9 프롬프트 그대로) → 비슷한 사례(reaction.predict) → 검색 (chain.retrieve의 예상 경로와 같은 범위)
- 붙여 넣은 기획서는 저장·임베딩하지 않는다. 검색어(항목 문장) 임베딩만 질문처럼 쓴다 (app 조회 전용)
- 챗봇 화면과 평가 하네스(eval/plan_eval.py)가 이 모듈을 함께 부른다
확인 실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m rag.plan_check eval/plans/plan_A.md
"""

import json, re, sys, unicodedata
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

from rag.chain import SCHEME
from rag.features import MODEL, extract
from rag.reaction import predict
from rag.retriever import QUALIFIED, KBRetriever, to_dict

load_dotenv()

PROMPT = """너는 광고·홍보 기획서에서 원문을 골라 그대로 복사한다. 고쳐 쓰거나 요약하지 않는다. JSON 객체 하나만 출력한다.
- common: 기획서 전체에 해당하는 정보가 적힌 원문 부분. 없으면 null
  · company: 회사·기관·브랜드가 무엇을 하는 곳인지 적힌 부분
  · disclosure: AI를 썼다는 사실을 알리는지·알리지 않는지에 대한 방침이 적힌 부분
  · period: 광고·캠페인을 집행하는 시기가 적힌 부분
- items: 이 기획에서 만들 제작물 중 AI(생성형 AI)를 쓰는 항목만. 제작물 하나가 항목 하나
  · AI를 쓰지 않는 제작물(실제 촬영, 사람·업체가 직접 만드는 것)은 고르지 않는다
  · quote: 그 항목을 설명하는 원문 문장 하나
출력 형식: {"common": {"company": "...", "disclosure": "...", "period": "..."}, "items": [{"quote": "..."}]}"""
QUESTION = "{common}\n이 기획의 항목: {quote}\n이 항목을 이렇게 만들면 소비자 반응은 어떨까?"   # exp_plan_common: 코드가 질문을 조립


@lru_cache(maxsize=1)
def _client():
    return OpenAI(default_headers={"Accept-Encoding": "gzip, deflate"})   # 키는 .env의 OPENAI_API_KEY


def _flat(t):
    """원문 대조용: NFC, 표 구분선(|) 제거, 공백 정리"""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(t)).replace("|", " ")).strip()


def extract_items(text):
    """기획서 → [{quote, question, common}]. LLM은 원문 인용만, 질문은 코드가 조립.
    공통 정보 인용이 원문에 없으면 버림 (지어낸 정보가 질문에 들어가지 않게). 형식 오류는 그대로 냄"""
    r = _client().chat.completions.create(model=MODEL, temperature=0, response_format={"type": "json_object"},
                                          messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text}])
    raw = json.loads(r.choices[0].message.content)
    src = _flat(text)
    common = {k: v for k, v in (raw.get("common") or {}).items()
              if k in ("company", "disclosure", "period") and isinstance(v, str) and v.strip() and _flat(v) in src}
    head = "\n".join(f"- {v.strip()}" for v in common.values())
    head = f"기획서 공통 정보:\n{head}" if head else ""
    return [{"quote": str(x["quote"]), "question": QUESTION.format(common=head, quote=str(x["quote"])).strip(),
             "common": json.dumps(common, ensure_ascii=False)} for x in raw["items"]]


def base_docs(q, k=10):
    """A: 범위를 좁히지 않은 검색 (탈락 사례만 제외 = retrieve()에서 비슷한 사례 0개일 때와 같음)"""
    return [to_dict(d) for d in KBRetriever(k=k, scheme=SCHEME, where=QUALIFIED).invoke(q)]


def item_docs(q, features, k=10):
    """B: 특징 → 비슷한 자격 사례 안에서만 검색 (retrieve()의 예상 경로와 같은 범위)"""
    pred = predict(features)
    ids = [s["case_id"] for s in pred["similar"]]
    where = {"case_id": {"$in": ids}} if ids else QUALIFIED
    return [to_dict(d) for d in KBRetriever(k=k, scheme=SCHEME, where=where).invoke(q)], pred


def analyze(text):
    """화면용: 항목마다 질문·특징·등급·비슷한 사례 (답변은 화면에서 chain.prepare(question, info=…)로)"""
    out = []
    for it in extract_items(text):
        f = extract(it["question"])["features"]
        p = predict(f)
        out.append({**it, "features": f, "grade": p["grade"], "similar": [s["case_id"] for s in p["similar"]],
                    "info": {"intent": "predict", "features": f}})   # 답변 때 chain.prepare(info=)로 같은 특징을 넘김
    return out


if __name__ == "__main__":
    for x in analyze(open(sys.argv[1], encoding="utf-8").read()):
        print(json.dumps(x, ensure_ascii=False))