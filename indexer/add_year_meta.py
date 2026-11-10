"""모든 청크 메타에 발행 연도(year, 정수)만 더한다 — 기간 필터용 (indexer 단계, 임베딩·본문은 그대로 → 검색 기준선 영향 없음)
Chroma는 크기 비교($gte·$lte)를 숫자에만 할 수 있어서, 문자열 published_at에서 앞 4자리를 숫자로 뽑아 둔다
실행 (프로젝트 루트): python -m indexer.add_year_meta"""
from collections import Counter
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
col = chromadb.PersistentClient(path=str(ROOT / "store" / "chroma")).get_collection("kb_kure_v1")
got = col.get(include=["metadatas"])
ids, metas, years, missing = [], [], Counter(), Counter()
for i, m in zip(got["ids"], got["metadatas"]):
    y = str(m.get("published_at") or "")[:4]
    if y.isdigit():
        ids.append(i)
        metas.append({**m, "year": int(y)})            # 기존 메타는 그대로, 키 하나만 추가
        years[(m.get("authority"), int(y))] += 1
    else:
        missing[m.get("doc_type")] += 1
for s in range(0, len(ids), 200):                       # 한 번에 너무 많이 보내지 않도록 나눠서
    col.update(ids=ids[s:s + 200], metadatas=metas[s:s + 200])
print(f"year 추가 {len(ids)}개 / 전체 {len(got['ids'])}개, 발행일 없음 {dict(missing) or 0}")
for (a, y), n in sorted(years.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
    print(f"  {a} {y}: {n}개")
