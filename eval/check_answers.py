"""
대표 질문 답변 확인 (Day 4 오전 4번): answer()를 유형별 대표 문항에 돌려 사람이 읽고 점검할 md를 만든다 (채점은 Day 5)
점검: 등급·수치가 코드 값 그대로인지, 출처 번호가 맞는지, 거절·약한 거절 문구, 자료에 없을 때 "확인할 수 없음"
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 python -m eval.check_answers --tag day4_answers_v1 [--ids C01,C04] [--set eval/probes/extra_compare.jsonl]
결과: report/evidence/<tag>.md (답변 원문), eval/results/<tag>.json (문항별 기록)
"""

import argparse, json
from pathlib import Path

from rag.chain import answer

ROOT = Path(__file__).resolve().parents[1]
IDS = ["P02", "P01", "P05", "F01", "C01", "C03", "C04", "C05", "G01", "G03", "S01", "U01", "U04"]   # 비교 4문항: 특징 값(C01·C04)·이름(C03·C05)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--ids", default="", help="쉼표로 문항 골라 돌리기 (예: C01,C04). 비우면 전체")
ap.add_argument("--set", default="eval/eval_set.jsonl", help="문항 파일 (평가셋 밖 확인 질문용). 평가셋이 아니면 --ids 없이 전체")
a = ap.parse_args()
if a.ids:
    IDS = [i.strip() for i in a.ids.split(",") if i.strip()]

items = {x["id"]: x for x in map(json.loads, filter(str.strip, open(ROOT / a.set, encoding="utf-8")))}
if a.set != "eval/eval_set.jsonl" and not a.ids:
    IDS = list(items)
md, log = [f"# {a.tag}\n"], []
for i in IDS:
    x = items[i]
    out = answer(x["q"])
    print(i, out["scope"], out["refusal"])
    md += [f"## {i} ({x['type']})", f"**질문** {x['q']}", f"**정답 기준** {x.get('answer_key', '')}",
           f"**판정** 범위 {out['scope']} · 거절 {out['refusal']} · 의도 {(out['retrieval'] or {}).get('intent')} · 비교 {out.get('compare')}",
           "```", out["text"], "```", "**출처**"]
    md += [f"- [{s['번호']}] {s['종류']} · {s['제목']} · {s['발행일']} {s['쪽']}" for s in out["sources"]] or ["- 없음"]
    if out.get("case_log"):
        md.append("**사례별 인용** " + ", ".join(f"[{g['case']}] {g['quote_src']}{g['problems'] or ''}" for g in out["case_log"]))
    if out.get("dropped"):
        md.append("**지운 문장** " + " / ".join(f"({d['why']}) {d['line']}" for d in out["dropped"]))
    if out.get("cite_problems"):
        md.append(f"**없는 출처 번호** {out['cite_problems']}")
    md.append("")
    log.append({"id": i, "scope": out["scope"], "refusal": out["refusal"], "text": out["text"],
                "case_log": out.get("case_log", []), "cite_problems": out.get("cite_problems", []),
                "compare": out.get("compare"), "dropped": out.get("dropped", []),
                "source_ids": [c["id"] for s in out["sources"] for c in s["청크"]]})

(ROOT / "report" / "evidence").mkdir(parents=True, exist_ok=True)
(ROOT / f"report/evidence/{a.tag}.md").write_text("\n".join(md), encoding="utf-8")
(ROOT / "eval/results").mkdir(parents=True, exist_ok=True)
json.dump(log, open(ROOT / f"eval/results/{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n→ report/evidence/{a.tag}.md")