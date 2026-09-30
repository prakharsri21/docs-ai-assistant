from app.rag.basic_rag import BasicRAG


def main() -> None:
    rag = BasicRAG()

    questions = [
        "What is gradient descent?",
        "What happens when the batch size is 1 in SGD?",
        "How does self-attention work in transformers?",
    ]

    for question in questions:
        print()
        print("=" * 80)
        print("QUESTION")
        print("=" * 80)
        print(question)
        print()

        result = rag.answer(
            question,
            top_k=5,
        )

        print("ANSWER")
        print("-" * 80)
        print(result["answer"])
        print()

        print("SOURCES")
        print("-" * 80)

        for source in result["sources"]:
            print(
                f"{source['source_file']} | "
                f"page {source['page']} | "
                f"{source['chunk_id']} | "
                f"similarity={source['similarity']:.4f}"
            )


if __name__ == "__main__":
    main()