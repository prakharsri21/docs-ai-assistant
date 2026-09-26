from __future__ import annotations

from sqlalchemy import create_engine, text

from app.core.config import settings
from app.embeddings.bge_m3 import create_embedding_model


def main() -> None:
    query = "What happens when the batch contains only one training example?"

    target_chunk = "deep_learning_lec2-p9-c01"

    embedding_model = create_embedding_model()
    query_embedding = embedding_model.embed_query(query)

    engine = create_engine(settings.DATABASE_URL)

    sql = text(
        """
        SELECT
            chunk_id,
            source_file,
            page,
            token_count,
            1 - (embedding <=> CAST(:embedding AS vector))
                AS similarity
        FROM rag_chunks
        WHERE chunk_id = :target_chunk
        """
    )

    with engine.connect() as connection:
        result = connection.execute(
            sql,
            {
                "embedding": str(query_embedding),
                "target_chunk": target_chunk,
            },
        ).mappings().first()

    print()
    print("=" * 70)
    print("TARGET CHUNK DIAGNOSTIC")
    print("=" * 70)

    if result is None:
        print("Target chunk not found.")
        return

    print(f"Chunk      : {result['chunk_id']}")
    print(f"Document   : {result['source_file']}")
    print(f"Page       : {result['page']}")
    print(f"Tokens     : {result['token_count']}")
    print(f"Similarity : {result['similarity']:.4f}")


if __name__ == "__main__":
    main()