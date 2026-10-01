from __future__ import annotations

from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """
    Cross-encoder reranker.

    Takes a query and candidate documents, scores each
    query/document pair jointly, and returns candidates
    ordered by relevance.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        self.model = CrossEncoder(
            model_name,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 10,
    ) -> list[dict]:

        if not results:
            return []

        pairs = []

        for result in results:
            document_text = self._build_document_text(
                result
            )

            pairs.append(
                [query, document_text]
            )

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for result, score in zip(
            results,
            scores,
        ):
            item = dict(result)

            item["reranker_score"] = float(
                score
            )

            reranked.append(item)

        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]

    @staticmethod
    def _build_document_text(
        result: dict,
    ) -> str:

        title = result.get("title", "")
        summary = result.get("summary", "")
        content = result.get("content", "")
        text = result.get("text", "")

        return "\n".join(
            part
            for part in [
                f"Title: {title}" if title else "",
                f"Summary: {summary}" if summary else "",
                f"Content: {content}" if content else "",
                text,
            ]
            if part
        )