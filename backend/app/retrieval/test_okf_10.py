from app.retrieval.okf_retriever import OKFRetriever


QUESTIONS = [
    "What happens when the batch size is one in stochastic gradient descent?",
    "How does self-attention allow a model to capture relationships between tokens?",
    "Why can training loss be different from test loss?",
    "What is the purpose of a learning rate during optimization?",
    "How does the perceptron update its parameters when it makes a mistake?",
    "What is the role of support vectors in a maximum-margin classifier?",
    "Why can smaller batches produce noisy gradient estimates?",
    "What is the purpose of regularization in machine learning?",
    "How does boosting improve a weak classifier?",
    "What is the difference between training on a batch and training on the full dataset?",
]


if __name__ == "__main__":

    retriever = OKFRetriever()

    print("=" * 100)
    print("OKF 10-QUESTION RETRIEVAL BENCHMARK")
    print("=" * 100)

    for i, question in enumerate(QUESTIONS, start=1):

        print("\n" + "=" * 100)
        print(f"QUESTION {i}")
        print("=" * 100)

        print(question)

        results = retriever.retrieve(
            question,
            top_k=5,
            candidate_k=50,
        )

        for rank, result in enumerate(results, start=1):

            print(
                f"\n#{rank} "
                f"[RRF={result.get('rrf_score', 0):.6f}]"
            )

            print(
                f"Title : {result['title']}"
            )

            print(
                f"Chunk : {result['chunk_id']}"
            )

            print(
                f"Source: "
                f"{result['source_file']} "
                f"page {result['page']}"
            )

            print(
                f"By    : "
                f"{result.get('retrieved_by')}"
            )

            print(
                f"Summary: "
                f"{result['summary'][:300]}"
            )