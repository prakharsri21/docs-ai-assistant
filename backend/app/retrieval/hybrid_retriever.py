from __future__ import annotations

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:

    def __init__(self) -> None:
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()

    def retrieve(
    self,
    query: str,
    top_k: int = 5,
    candidate_k: int = 50,
) -> list[dict]:

        dense_results = self.dense.retrieve(
        query,
        top_k=candidate_k,
    )

        bm25_results = self.bm25.retrieve(
        query,
        top_k=candidate_k,
    )

        fused_results = reciprocal_rank_fusion(
        {
            "dense": dense_results,
            "bm25": bm25_results,
        }
    )

        return fused_results[:top_k]