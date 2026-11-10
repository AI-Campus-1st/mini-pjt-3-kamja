"""
실패 문항 진단: top3 청크 내용과 정답 청크(가장 점수 높은 것)의 점수·순위를 나란히 출력 (LLM 호출 없음, 조회만)
실행: HF_HUB_OFFLINE=1 python -m eval.inspect_fail --tag run1_chain > report/evidence/fail_run1_chain.txt
"""

import argparse, json, re
from pathlib import Path

import pandas as pd
from rag.retriever import _col, search

ROOT = Path(__file__).resolve().parents[1]
short = lambda t: " ".join(t.split())[:150]


def where_str(m):
    return f'{m.get("doc_id") or m.get("case_id")} p{m["page_start"]}-{m["page_end"]}' if "page_start" in m else (m.get("doc_id") or m.get("case_id"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
    det = pd.read_csv(next((ROOT / "eval" / "results").rglob(f"{a.tag}_detail.csv")), encoding="utf-8-sig")
    items = {x["id"]: x for x in map(json.loads, filter(str.strip, (ROOT / "eval" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()))}
    fail = det[(det.hit3 == 0) & (det.type != "unanswerable")]

    for r in fail.itertuples():
        x = items[r.id]
        print(f"\n===== {r.id} ({r.type}) 전체 순위 {r.rank} =====\nQ: {x['q']}")
        print(f"정답: pages={x.get('gold_pages')} cases={x.get('gold_case_ids')}")
        ids = re.findall(r"([^\s|]+)\(", r.top3)
        got = _col().get(ids=ids, include=["documents", "metadatas"])
        by = dict(zip(got["ids"], zip(got["documents"], got["metadatas"])))
        for i, cid in enumerate(ids, 1):
            t, m = by[cid]
            print(f"  top{i} [{m['doc_type']}] {where_str(m)} | {short(t)}")
        for d, pages in (x.get("gold_pages") or {}).items():   # 그 문서 안에서 정답 쪽 청크의 점수·순위
            res = search(x["q"], k=200, where={"doc_id": d})
            hit = next(((j, h) for j, h in enumerate(res, 1) if any(h["meta"]["page_start"] <= p <= h["meta"]["page_end"] for p in pages)), None)
            if hit:
                print(f"  정답 청크 ({d} 안 {hit[0]}위, 점수 {hit[1]['score']}) {where_str(hit[1]['meta'])} | {short(hit[1]['text'])}")
            else:
                print(f"  정답 청크 없음: {d} {pages} 쪽을 덮는 청크가 없음 (청킹 확인 필요)")
        if x.get("gold_case_ids"):
            res = search(x["q"], k=1, where={"case_id": {"$in": x["gold_case_ids"]}})
            if res:
                print(f"  정답 사례 최고 청크 (점수 {res[0]['score']}) {res[0]['id']} | {short(res[0]['text'])}")


if __name__ == "__main__":
    main()
