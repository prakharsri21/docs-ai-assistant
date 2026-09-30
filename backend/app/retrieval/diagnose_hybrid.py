from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.rrf import reciprocal_rank_fusion


QUERY = (
    "What happens when the batch "
    "contains only one training example?"
)

TARGET = "deep_learning_lec2-p9-c01"


def main() -> None:
    dense = DenseRetriever()
    bm25 = BM25Retriever()

    dense_results = dense.retrieve(
        QUERY,
        top_k=50,
    )

    bm25_results = bm25.retrieve(
        QUERY,
        top_k=50,
    )

    hybrid_results = reciprocal_rank_fusion(
        {
            "dense": dense_results,
            "bm25": bm25_results,
        }
    )

    target_rank = None
    target_result = None

    for rank, result in enumerate(
        hybrid_results,
        start=1,
    ):
        if result["chunk_id"] == TARGET:
            target_rank = rank
            target_result = result
            break

    print()
    print("=" * 80)
    print("TARGET HYBRID RANK")
    print("=" * 80)

    print(f"Query: {QUERY}")
    print()

    print(f"Target chunk : {TARGET}")

    if target_result is None:
        print("Hybrid rank : NOT FOUND")
        return

    print(f"Hybrid rank  : {target_rank}")
    print(
        f"RRF score    : "
        f"{target_result['rrf_score']:.5f}"
    )
    print(
        f"Retrieved by : "
        f"{target_result['retrieved_by']}"
    )
    print(
        f"Ranks        : "
        f"{target_result['retriever_ranks']}"
    )


if __name__ == "__main__":
    main()