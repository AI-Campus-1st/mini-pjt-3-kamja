"""
비슷한 사례 판정 + 예상 등급을 코드 규칙으로 계산 (기획서 2-5). store는 읽기만 한다
- 비슷한 사례: 특징 4개 중 2개 이상 일치 (빈 값은 불일치), 자격 사례(qualified=1)만
- 등급 (축소 기준 적용): 비슷한 사례 2개 이상 = 관찰됨, 2개 미만 = 근거 부족. "높음"은 쓰지 않음
- 주된 이유는 사례 요약의 stats(최상위 댓글 기준, 5개 미만이면 None)를 그대로 쓴다
확인 실행 (평가셋 정답 특징으로 규칙만 검증, LLM 없음): python -m rag.reaction
"""

import json, sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rag.retriever import _col

ROOT = Path(__file__).resolve().parents[1]
FEATS, MIN_MATCH, MIN_CASES = ["industry", "ai_part", "disclosed", "period"], 2, 2


@lru_cache(maxsize=1)
def cases():
    """사례 요약 청크 메타데이터(특징·자격) + docs.db case_summary.stats (읽기 전용으로 엶)"""
    metas = _col().get(where={"doc_type": "case_summary"}, include=["metadatas"])["metadatas"]
    with sqlite3.connect(f"file:{ROOT / 'store' / 'docs.db'}?mode=ro", uri=True) as con:
        stats = {cid: json.loads(s) for cid, s in con.execute("SELECT case_id, stats FROM case_summary")}
    return {m["case_id"]: {**{f: m.get(f) for f in FEATS}, "qualified": m["qualified"], "stats": stats[m["case_id"]]}
            for m in metas}


def match(features):
    """특징 2개 이상 일치하는 사례 (자격 무관). 일치 수가 많은 순"""
    out = []
    for cid, c in cases().items():
        n = sum(1 for f in FEATS if features.get(f) and c[f] and features[f] == c[f])
        if n >= MIN_MATCH:
            out.append({"case_id": cid, "n_match": n, "qualified": c["qualified"]})
    return sorted(out, key=lambda x: (-x["n_match"], x["case_id"]))


def predict(features, allowed=None):
    """allowed: 필터에 맞는 사례 id 집합 (None = 제한 없음, 기본 동작 그대로)"""
    m = match(features)
    similar = []
    for x in (x for x in m if x["qualified"] and (allowed is None or x["case_id"] in allowed)):
        st = cases()[x["case_id"]]["stats"]
        similar.append({**x, **{k: st[k] for k in ["top_related", "pos_pct", "neg_pct", "neg_main", "pos_main"]}})
    res = {"grade": "관찰됨" if len(similar) >= MIN_CASES else "근거 부족", "similar": similar,
           "excluded": [x["case_id"] for x in m if not x["qualified"]],   # 특징은 맞지만 탈락한 사례 (근거 아님)
           "filtered": [x["case_id"] for x in m if x["qualified"] and allowed is not None and x["case_id"] not in allowed]}
    for d in ["neg", "pos"]:   # 주된 이유별로 몇 개 사례에서 1위였는지
        cnt = Counter(s[f"{d}_main"]["code"] for s in similar if s[f"{d}_main"]).most_common()
        res[f"{d}_reasons"] = dict(cnt)
        res[f"{d}_tie"] = len(cnt) > 1 and cnt[0][1] == cnt[1][1]   # 1·2위 사례 수가 같으면 동률 (2-5와 같은 원칙)
    return res


if __name__ == "__main__":
    lines = (ROOT / "eval" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()
    for x in (x for x in map(json.loads, filter(str.strip, lines)) if x["type"] == "predict"):
        r = predict(x["features"])
        ids = sorted(s["case_id"] for s in r["similar"])
        print(x["id"], "사례", "OK" if ids == sorted(x["gold_case_ids"]) else f"X {ids}",
              "| 등급", "OK" if r["grade"] == x["expected_grade"] else f'X {r["grade"]}', "| 부정 이유", r["neg_reasons"], "(동률)" if r["neg_tie"] else "")