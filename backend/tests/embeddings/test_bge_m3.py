from .bge_m3 import create_embedding_model


def main() -> None:
    embeddings = create_embedding_model()

    texts = [
        "What is gradient descent?",
        "Transformers use self-attention mechanisms.",
    ]

    vectors = embeddings.embed_documents(texts)

    print(f"Number of vectors: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")
    print(f"First 5 values: {vectors[0][:5]}")

    query = "How does gradient descent work?"
    query_vector = embeddings.embed_query(query)

    print(f"Query vector dimension: {len(query_vector)}")


if __name__ == "__main__":
    main()