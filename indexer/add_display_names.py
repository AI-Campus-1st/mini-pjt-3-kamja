"""사례 요약 청크 메타에 화면 표시용 사례명(display_name)만 더한다 (indexer 단계, 임베딩·본문은 그대로 → 검색 기준선 영향 없음)
실행 (프로젝트 루트): python -m indexer.add_display_names"""
import unicodedata
from pathlib import Path

import chromadb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
nfc = lambda s: unicodedata.normalize("NFC", str(s)).strip()

cases = pd.read_csv(ROOT / "collector" / "cases.csv", encoding="utf-8-sig", dtype=str).fillna("")
names = {nfc(r.case_id): nfc(r.display_name) for r in cases.itertuples() if nfc(r.display_name)}
col = chromadb.PersistentClient(path=str(ROOT / "store" / "chroma")).get_collection("kb_kure_v1")
got = col.get(where={"doc_type": "case_summary"}, include=["metadatas"])
ids, metas = [], []
for i, m in zip(got["ids"], got["metadatas"]):
    if m.get("case_id") in names:
        ids.append(i)
        metas.append({**m, "display_name": names[m["case_id"]]})   # 기존 메타는 그대로 두고 키 하나만 추가
col.update(ids=ids, metadatas=metas)
print(f"사례 요약 {len(ids)}개에 display_name 추가")
for i, m in zip(ids, metas):
    print(f"  {m['case_id']}: {m['display_name']}")
