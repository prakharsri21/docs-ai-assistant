from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker


QUERY = (
    "What happens when the batch size is one "
    "in stochastic gradient descent?"
)


if __name__ == "__main__":

    print("=" * 80)
    print("DAY 11 — RERANKER TEST")
    print("=" * 80)

    retriever = HybridRetriever()

    candidates = retriever.retrieve(
        QUERY,
        top_k=50,
    )

    print(
        f"\nRetrieved candidates: {len(candidates)}"
    )

    reranker = Reranker()

    results = reranker.rerank(
        QUERY,
        candidates,
        top_k=10,
    )

    print("\n" + "=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(f"\n#{rank}")

        print(
            "Chunk :",
            result["chunk_id"],
        )

        print(
            "Source:",
            result["source_file"],
            "page",
            result["page"],
        )

        print(
            "RRF   :",
            f"{result.get('rrf_score', 0):.6f}",
        )

        print(
            "Rerank:",
            f"{result['reranker_score']:.6f}",
        )

        print(
            "Title :",
            result.get("title", "")
            or result.get("text", "")[:100],
        )