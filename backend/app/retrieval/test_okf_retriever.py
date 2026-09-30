from app.retrieval.okf_retriever import OKFRetriever


QUERY = (
    "What happens when the batch contains "
    "only one training example?"
)


if __name__ == "__main__":

    retriever = OKFRetriever()

    print("=" * 80)
    print("OKF RETRIEVAL TEST")
    print("=" * 80)

    print("\nQuery:")
    print(QUERY)

    results = retriever.retrieve(
        QUERY,
        top_k=10,
        candidate_k=50,
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n#{rank}"
        )

        print(
            f"Title       : {result['title']}"
        )

        print(
            f"Chunk       : {result['chunk_id']}"
        )

        print(
            f"Source      : "
            f"{result['source_file']} "
            f"page {result['page']}"
        )

        print(
            f"RRF score   : "
            f"{result.get('rrf_score', 0):.6f}"
        )

        print(
            f"Retrieved by: "
            f"{result.get('retrieved_by')}"
        )

        print(
            f"Ranks       : "
            f"{result.get('retriever_ranks')}"
        )

        print(
            f"Summary     : "
            f"{result['summary'][:250]}"
        )