from app.retrieval.hybrid_retriever import (
    HybridRetriever,
)


def main() -> None:

    retriever = HybridRetriever()

    query = (
        "What happens when the batch "
        "contains only one training example?"
    )

    results = retriever.retrieve(
        query,
        top_k=10,
        candidate_k=10,
    )

    print()
    print("=" * 80)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 80)
    print(f"Query: {query}")
    print()

    for rank, result in enumerate(
    results,
    start=1,
    ):

        print(
        f"{rank:2}. "
        f"RRF={result['rrf_score']:.5f} | "
        f"{result['source_file']} | "
        f"p.{result['page']} | "
        f"{result['chunk_id']}"
    )

        print(
        f"Retrieved by: "
        f"{result['retrieved_by']}"
    )

        print(
        f"Ranks: "
        f"{result['retriever_ranks']}"
    )

        print(
        result["text"][:400]
        .replace("\n", " ")
    )

        print()


if __name__ == "__main__":
    main()