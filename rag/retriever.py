"""
Chroma 조회 전용 검색기 (기획서 4-1). 질문만 임베딩하고 인덱스에는 쓰지 않는다.
- scheme: 댓글 청크 안 ("reason" / "time"). 비댓글 청크(common)는 항상 함께 검색
- KBRetriever: search()를 LangChain 검색기로 감쌈 (결과·필터는 search()와 같음)
- where: 추가 필터 (예: {"doc_type": "gov"}). scheme 조건과 $and로 묶음
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import chromadb
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
COLLECTION, MODEL = "kb_kure_v1", "nlpai-lab/KURE-v1"
QUALIFIED = {"$or": [{"qualified": 1}, {"doc_type": {"$in": ["gov", "news_provided"]}}]}


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(MODEL)          # 처음 한 번만 불러옴 (약 12초)


@lru_cache(maxsize=1)
def _col():
    return chromadb.PersistentClient(path=str(ROOT / "store" / "chroma")).get_collection(COLLECTION)


def search(q, k=3, scheme="time", where=None, qualified_only=False):
    cond = {"scheme": {"$in": ["common", scheme]}}
    if qualified_only:   # 예상 질문: 탈락 사례(qualified=0) 제외. 정부·뉴스는 qualified 키가 없어 따로 허용
        where = {"$and": [where, QUALIFIED]} if where else QUALIFIED
    if where:
        cond = {"$and": [cond, where]}
    emb = _model().encode([q], normalize_embeddings=True).tolist()
    r = _col().query(query_embeddings=emb, n_results=k, where=cond,
                     include=["documents", "metadatas", "distances"])
    return [{"id": i, "text": t, "meta": m, "score": round(1 - d, 4)}   # 코사인 거리 → 유사도
            for i, t, m, d in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]


class KBRetriever(BaseRetriever):
    """search()를 그대로 부르는 LangChain 검색기. 청크 ID·유사도는 metadata의 _id·_score로 전달"""
    k: int = 3
    scheme: str = "time"
    where: Optional[dict] = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [Document(page_content=r["text"], metadata={**r["meta"], "_id": r["id"], "_score": r["score"]})
                for r in search(query, k=self.k, scheme=self.scheme, where=self.where)]


def to_dict(d):
    """Document → 평가 하네스가 쓰던 {id, text, meta, score} 모양"""
    meta = {k: v for k, v in d.metadata.items() if not k.startswith("_")}
    return {"id": d.metadata["_id"], "text": d.page_content, "meta": meta, "score": d.metadata["_score"]}


def get_summaries(case_ids):
    """사례 요약 청크를 case_id로 조회 (검색 아님, 읽기만). 예상 답변의 [근거 사례] 출처용"""
    if not case_ids:
        return []
    r = _col().get(where={"$and": [{"doc_type": "case_summary"}, {"case_id": {"$in": list(case_ids)}}]},
                   include=["documents", "metadatas"])
    found = {m["case_id"]: {"id": i, "text": t, "meta": m, "score": None}
             for i, t, m in zip(r["ids"], r["documents"], r["metadatas"])}
    return [found[c] for c in case_ids if c in found]   # 비슷한 사례 순서 유지