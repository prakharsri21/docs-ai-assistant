from __future__ import annotations

from sqlalchemy import create_engine, text

from app.core.config import settings
from app.embeddings.bge_m3 import create_embedding_model


TOP_K = 5


class DenseRetriever:
    def __init__(self) -> None:
        self.embedding_model = create_embedding_model()
        self.engine = create_engine(settings.DATABASE_URL)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Retrieve the most semantically similar chunks
        using BGE-M3 dense embeddings.
        """

        query_embedding = self.embedding_model.embed_query(query)

        sql = text(
            """
            SELECT
                chunk_id,
                parent_page_id,
                domain,
                document_id,
                source_file,
                page,
                chunk_index,
                token_count,
                has_visual,
                ocr_candidate,
                text,
                1 - (embedding <=> CAST(:embedding AS vector))
                    AS similarity
            FROM rag_chunks
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
            """
        )

        with self.engine.connect() as connection:
            rows = connection.execute(
                sql,
                {
                    "embedding": str(query_embedding),
                    "top_k": top_k,
                },
            ).mappings().all()

        return [dict(row) for row in rows]


def main() -> None:
    retriever = DenseRetriever()

    query = "What is gradient descent?"

    results = retriever.retrieve(query, top_k=5)

    print()
    print("=" * 70)
    print("DENSE RETRIEVAL TEST")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print(f"Similarity : {result['similarity']:.4f}")
        print(f"Document   : {result['source_file']}")
        print(f"Page       : {result['page']}")
        print(f"Chunk      : {result['chunk_id']}")
        print(f"Tokens     : {result['token_count']}")
        print()
        print(result["text"][:600])
        print()


if __name__ == "__main__":
    main()