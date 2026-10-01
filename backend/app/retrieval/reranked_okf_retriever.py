from __future__ import annotations

from app.retrieval.okf_retriever import OKFRetriever
from app.retrieval.reranker import Reranker


class RerankedOKFRetriever:

    def __init__(
        self,
        candidate_k: int = 50,
        top_k: int = 10,
    ):
        self.retriever = OKFRetriever()
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
            candidate_k=self.candidate_k,
        )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=final_k,
        )