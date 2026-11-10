"""
기획서 확인 탭 전용 피드백: 가상 소비자 한마디(한 문장) + 이런 반응이 나올 이유(문장마다 근거 사례 번호)
- 방향(부정/긍정)은 코드가 정함: 비슷한 사례 중 그 방향 댓글 비율이 더 높은 사례가 많은 쪽, 같으면 부정 (_quote_plan과 같은 규칙)
- 관찰됨일 때만 만든다. 근거 부족이면 LLM을 부르지 않음 (사례 하나로 대사를 지어내면 과장)
- LLM에는 코드가 계산한 주된 이유와 검사를 통과한 실제 인용 댓글만 넘김. 결과는 코드가 검사:
  근거 번호 없음·없는 번호·숫자·등급 단어 → 그 문장 지움. 번호는 코드가 다시 붙임
- 챗봇 탭·평가 하네스는 이 모듈을 쓰지 않음 (기준선 영향 없음). 정량 측정 없음 (리포트에 정성 확인만으로 기록)
"""

import json, re
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

from rag.chain import BAN_WORDS, LLM, NUM

load_dotenv()

MAX_PERSONA, MAX_LINE, MAX_FB, N_FB = 20, 60, 150, 3
DIGIT = re.compile(r"\d")
REF = re.compile(r"\s*\[\d+\]")
PROMPT = """너는 광고 기획자에게 소비자 반응 피드백을 주는 도우미다. 주어진 근거(비슷한 국내 사례의 실제 댓글과 주된 이유)만 써서 JSON 객체 하나로 답한다.
- persona: 이 기획 항목을 볼 법한 가상의 소비자 한 명 (나이대·관심사 정도, 15자 이내). 실존 인물·브랜드 이름은 쓰지 않는다
- line: 그 사람이 주어진 방향으로 할 법한 말 한 문장 (40자 이내, 구어체). 근거 댓글의 이유와 결이 같아야 하고, 근거 댓글을 그대로 베끼지 않는다
- feedback: 이런 반응이 왜 나올지 기획자에게 설명하는 문장 2~3개. 문장마다 근거 사례 번호(refs)를 하나 이상 단다
  · 근거에 있는 이유와 댓글 내용만 쓴다. 숫자·비율과 등급 단어(관찰됨, 근거 부족, 예상, 가능성, 전망)는 쓰지 않는다
  · 주된 이유가 'AI와 무관한 반응'이면, AI보다 광고 내용 자체가 반응을 좌우했다고 말한다
  · 마지막 문장은 기획에서 점검할 점 하나를 근거와 이어서 제안한다
출력 형식: {"persona": "...", "line": "...", "feedback": [{"text": "...", "refs": [1]}]}"""


@lru_cache(maxsize=1)
def _client():
    return OpenAI(default_headers={"Accept-Encoding": "gzip, deflate"})   # 키는 .env


def direction(pred):
    """코드가 대사 방향을 정함 → (방향, 화면에 보일 설명 한 줄)"""
    sim = pred["similar"]
    neg = sum(s["neg_pct"] >= s["pos_pct"] for s in sim)
    d, k = ("부정", neg) if neg >= len(sim) - neg else ("긍정", len(sim) - neg)
    tie = " (같은 수면 부정 쪽)" if neg * 2 == len(sim) else ""
    return d, f"비슷한 사례 {len(sim)}개 중 {k}개에서 {d} 댓글 비율이 더 높아 {d} 쪽 한마디를 골랐습니다{tie}."


def _evidence(r):
    """LLM에 넘길 근거: 사례별 주된 이유(코드 계산) + 검사를 통과한 실제 인용 댓글"""
    quotes = {g["ref"]: g for g in r.get("case_log", []) if g.get("quote")}
    out = []
    for row in r["cases"]:
        n = int(row["번호"].strip("[]"))
        line = f"[{n}] {row['사례']} — 부정 주된 이유: {row['부정 주된 이유']}, 긍정 주된 이유: {row['긍정 주된 이유']}"
        if n in quotes:
            line += f"\n    실제 댓글({quotes[n]['dir']}): \"{quotes[n]['quote']}\""
        out.append(line)
    return "\n".join(out)


def _bad(text):
    return bool(DIGIT.search(text) or NUM.search(text) or any(w in text for w in BAN_WORDS))


def check(raw, valid):
    """LLM 결과 → 화면에 쓸 값 + 지운 기록. valid: 쓸 수 있는 사례 번호"""
    log = []
    persona = str(raw.get("persona") or "").strip()
    if not persona or len(persona) > MAX_PERSONA:   # "20대 직장인"처럼 나이대 숫자는 허용
        log.append(f"페르소나 교체: {persona!r}")
        persona = "가상의 소비자"
    line = str(raw.get("line") or "").strip().strip('"“”')
    if not line or len(line) > MAX_LINE or _bad(line):
        log.append(f"한마디 지움: {line!r}")
        line = ""
    fb = []
    for x in raw.get("feedback") or []:
        text = REF.sub("", str((x or {}).get("text", ""))).strip()
        refs = sorted({int(n) for n in (x or {}).get("refs", []) if str(n).isdigit()})
        why = ("근거 번호 없음" if not refs else "없는 번호" if not set(refs) <= valid else
               "숫자·등급 단어" if _bad(text) else "너무 김" if len(text) > MAX_FB else "")
        if why or not text:
            log.append(f"피드백 지움({why or '빈 문장'}): {text}")
            continue
        fb.append(text + " " + "".join(f"[{n}]" for n in refs))
    return {"persona": persona, "line": line, "feedback": fb[:N_FB], "dropped": log}


def feedback(item, r):
    """item: plan_check.analyze의 항목 하나, r: chain.prepare + stream_body 결과. 관찰됨이 아니면 None"""
    pred = (r.get("retrieval") or {}).get("prediction")
    if r.get("refusal") or not pred or pred["grade"] != "관찰됨" or not r.get("cases"):
        return None
    d, why = direction(pred)
    msg = f"[기획 항목]\n{item['quote']}\n\n[한마디 방향]\n{d}\n\n[근거 사례]\n{_evidence(r)}"
    res = _client().chat.completions.create(model=LLM, temperature=0, response_format={"type": "json_object"},
                                            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": msg}])
    raw = json.loads(res.choices[0].message.content)
    valid = {int(row["번호"].strip("[]")) for row in r["cases"]}
    return {"dir": d, "dir_note": why, **check(raw, valid), "raw": raw}
