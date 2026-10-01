from app.retrieval.okf_retriever import OKFRetriever


QUERY = (
    "What happens when the batch contains "
    "only one training example?"
)

TARGET = "deep_learning_lec2-p9-c01"


if __name__ == "__main__":

    retriever = OKFRetriever()

    dense = retriever._dense_retrieve(
        QUERY,
        top_k=50,
    )

    bm25 = retriever._bm25_retrieve(
        QUERY,
        top_k=50,
    )

    print("=" * 80)
    print("OKF DIAGNOSTIC")
    print("=" * 80)

    dense_rank = next(
        (
            r["dense_rank"]
            for r in dense
            if r["chunk_id"] == TARGET
        ),
        None,
    )

    bm25_rank = next(
        (
            r["bm25_rank"]
            for r in bm25
            if r["chunk_id"] == TARGET
        ),
        None,
    )

    print("\nTarget:")
    print(TARGET)

    print("\nTarget ranks:")
    print("  OKF dense:", dense_rank)
    print("  OKF BM25 :", bm25_rank)

    print("\nTarget OKF record:")

    target_node = next(
        (
            n
            for n in retriever.nodes
            if n["source"]["chunk_id"] == TARGET
        ),
        None,
    )

    if target_node:
        print("Title:")
        print(target_node["title"])

        print("\nSummary:")
        print(target_node["summary"])

        print("\nKeywords:")
        print(target_node["keywords"])

        print("\nRelationships:")
        print(target_node["relationships"])

    print("\nTop 10 OKF dense:")
    for r in dense[:10]:
        print(
            r["dense_rank"],
            r["chunk_id"],
            "|",
            r["title"],
        )

    print("\nTop 10 OKF BM25:")
    for r in bm25[:10]:
        print(
            r["bm25_rank"],
            r["chunk_id"],
            "|",
            r["title"],
        )