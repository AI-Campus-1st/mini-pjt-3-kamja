"""
스트리밍 확인 (Day 4 오후 2번): UI와 같은 순서(prepare → stream_body)로 돌려
① 화면에 흘러나온 글 = 최종 본문(out["body"])인지  ② 첫 글자까지·전체 걸린 시간을 기록한다
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m eval.check_stream --tag day4_stream_v1 [--ids P02,C01]
결과: eval/results/<tag>.json
"""

import argparse, json, time
from pathlib import Path

from rag.chain import prepare, stream_body

ROOT = Path(__file__).resolve().parents[1]
IDS = ["P02", "P01", "F01", "C01", "C04", "G01", "U04", "U01"]   # 예상·근거 부족·사실·비교 2·정부·범위 놓침 후보·거절

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--ids", default="")
a = ap.parse_args()
ids = [i.strip() for i in a.ids.split(",") if i.strip()] or IDS
items = {x["id"]: x for x in map(json.loads, open(ROOT / "eval/eval_set.jsonl", encoding="utf-8"))}

log, flat = [], lambda s: " ".join(s.split())
for i in ids:
    t0 = time.time()
    p = prepare(items[i]["q"])
    t_prep, first, shown, out = time.time() - t0, None, "", {}
    for piece in stream_body(p, out):
        if piece and first is None:
            first = time.time() - t0
        shown += piece
    body = out.get("body", "")
    same = flat(shown) == flat(body)
    print(f"{i}: 같음={same} · 준비 {t_prep:.1f}초 · 첫 글자 {first or 0:.1f}초 · 전체 {time.time() - t0:.1f}초 · 거절 {p['refusal']}")
    if not same:
        print("  화면:", shown[:200], "\n  최종:", body[:200])
    log.append({"id": i, "same": same, "prepare_s": round(t_prep, 1), "first_s": round(first, 1) if first else None,
                "total_s": round(time.time() - t0, 1), "refusal": p["refusal"], "shown": shown, "body": body,
                "sources_cleared": "sources" in out and not out["sources"]})

(ROOT / "eval/results").mkdir(parents=True, exist_ok=True)
json.dump(log, open(ROOT / f"eval/results/{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n같음 {sum(x['same'] for x in log)}/{len(log)} → eval/results/{a.tag}.json")
