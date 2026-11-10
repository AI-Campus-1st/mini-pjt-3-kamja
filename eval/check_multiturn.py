"""멀티턴 확인 (Day 4 오후 4번, 가이드 4-7 '이전 대화를 참조하는 질문이 정상 처리되는지'):
화면과 같은 순서(앞 질문 → prepare(q, history) )로 대화 4개를 돌려, 다시 쓴 질문에 필요한 말이 들어갔는지 코드로 확인
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m eval.check_multiturn --tag day4_multiturn_v1
결과: eval/results/<tag>.json"""
import argparse, json
from pathlib import Path

from rag.chain import prepare, stream_body
from rag.rewrite import summary

ROOT = Path(__file__).resolve().parents[1]
# (첫 질문, 이어지는 질문, 다시 쓴 질문에 있어야 할 말, 없어야 할 말, 기대: refusal/predict/compare/other)
CASES = [
    ("식품 광고 영상 전체를 AI로 만들고 'AI를 활용하여 광고를 만들었다'고 명시하면 소비자 반응은 어떨까?",
     "그럼 뷰티 업종이면?", ["뷰티", "영상 전체"], ["식품"], "predict"),
    ("AI로 만든 광고 이미지도 저작권 등록을 할 수 있나?",
     "그럼 사람이 AI 결과물을 많이 수정하면?", ["저작권", "수정"], [], "other"),
    ("페리카나 AI 광고의 소비자 반응은 어땠어?",
     "빙그레 캠페인이랑 비교하면?", ["페리카나", "빙그레"], [], "compare"),
    ("식품 광고 영상 전체를 AI로 만들고 'AI를 활용하여 광고를 만들었다'고 명시하면 소비자 반응은 어떨까?",
     "AI 광고를 쓰면 매출이 몇 % 떨어질지 예상해줘", ["매출"], [], "refusal"),   # 혼자 이해되는 질문 → 거의 그대로, 거절 유지
    # (v2 추가) 프롬프트 예시와 다른 값으로 '특징 하나 바꾸기'를 다시 확인 (예시에 맞춘 것만 통과하는지 보기)
    ("교육 광고에서 모델 인물만 AI로 만들고 AI 사용을 밝히지 않으면 반응은?",
     "금융 업종이라면?", ["금융", "모델"], ["교육"], "predict"),
]


def kind(p):
    return "refusal" if p["refusal"] else "predict" if p.get("is_predict") else "compare" if p.get("is_compare") else "other"


ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
a = ap.parse_args()
log = []
for first, follow, must, must_not, want in CASES:
    p1 = prepare(first)
    out = {}
    for _ in stream_body(p1, out):
        pass
    p2 = prepare(follow, history=[{"q": p1["question"], "a": summary({**p1, **out})}])
    q2 = p2["question"]
    ok_words = all(w in q2 for w in must) and not any(w in q2 for w in must_not)
    ok = ok_words and kind(p2) == want
    print(f"{'통과' if ok else '확인'} | {follow} → {q2}\n     처리 {kind(p2)} (기대 {want}) · 필요한 말 {'있음' if ok_words else '부족'}"
          + (f" · 오류 {p2['rewrite']['error']}" if p2["rewrite"]["error"] else ""))
    log.append({"first": first, "follow": follow, "rewritten": q2, "kind": kind(p2), "want": want,
                "words_ok": ok_words, "pass": ok, "head": p2["head"][:200], "error": p2["rewrite"]["error"]})
json.dump(log, open(ROOT / f"eval/results/{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n통과 {sum(x['pass'] for x in log)}/{len(log)} → eval/results/{a.tag}.json")