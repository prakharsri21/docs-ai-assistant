from __future__ import annotations

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker


class RerankedHybridRetriever:

    def __init__(
        self,
        candidate_k: int = 50,
        top_k: int = 10,
    ):
        self.retriever = HybridRetriever()
        self.reranker = Reranker()

        self.candidate_k = candidate_k
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:

        final_k = top_k or self.top_k

        candidates = self.retriever.retrieve(
            query,
            top_k=self.candidate_k,
        )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=final_k,
        )