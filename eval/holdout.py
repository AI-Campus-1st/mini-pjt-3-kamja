"""
숨겨 두기 검증 (기획서 6-2): 자격 사례 X를 빼고 나머지 사례로 X의 등급·주된 이유를 예상해 실제와 비교
- 질문 = X의 실제 특징 4개 (LLM 특징 추출 오류를 섞지 않고 예상 규칙만 측정)
- 사례 빼기 = predict(allowed=나머지 자격 사례). 인덱스는 다시 만들지 않음, store는 읽기만
- 엄격 일치: 예상 1위 이유가 단독 1위이고 실제와 같음 / 느슨 일치: 동률 1위 중 하나가 실제와 같음
- 실제 주된 이유가 없으면(그 방향 댓글 5개 미만) 비교 불가로 따로 셈
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m eval.holdout --tag day5_holdout_v1
"""

import argparse, json
from pathlib import Path

from rag.reaction import FEATS, cases, predict

ROOT = Path(__file__).resolve().parents[1]


def judge(res, actual, d):
    """d = neg/pos. 반환: (상태, 예상 1위 목록)"""
    if not actual:
        return "비교 불가", []
    cnt = res[f"{d}_reasons"]
    if res["grade"] == "근거 부족" or not cnt:
        return "근거 부족", []
    top = max(cnt.values())
    tops = [c for c, n in cnt.items() if n == top]
    if len(tops) == 1:
        return ("엄격 일치" if tops[0] == actual else "불일치"), tops
    return ("느슨 일치" if actual in tops else "불일치"), tops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    q_ids = sorted(cid for cid, c in cases().items() if c["qualified"])
    rows = []
    for x in q_ids:
        c = cases()[x]
        res = predict({f: c[f] for f in FEATS}, allowed=set(q_ids) - {x})
        row = {"case_id": x, "grade": res["grade"], "similar": [s["case_id"] for s in res["similar"]]}
        for d in ["neg", "pos"]:
            actual = (c["stats"].get(f"{d}_main") or {}).get("code")
            state, tops = judge(res, actual, d)
            row |= {f"{d}_actual": actual, f"{d}_pred": tops, f"{d}_state": state}
        rows.append(row)
        print(row)

    summary = {"tag": a.tag, "n_cases": len(rows),
               "insufficient_rate": round(sum(r["grade"] == "근거 부족" for r in rows) / len(rows), 3)}
    for d in ["neg", "pos"]:
        st = [r[f"{d}_state"] for r in rows]
        base = [s for s in st if s not in ("비교 불가", "근거 부족")]   # 실제 이유가 있고 예상도 한 사례
        strict = st.count("엄격 일치")
        summary[d] = {"counts": {s: st.count(s) for s in sorted(set(st))}, "n_judged": len(base),
                      "strict": round(strict / len(base), 3) if base else None,
                      "loose": round((strict + st.count("느슨 일치")) / len(base), 3) if base else None}
    summary["rows"] = rows

    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
