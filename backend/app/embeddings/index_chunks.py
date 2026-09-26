from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.config import settings
from app.embeddings.bge_m3 import create_embedding_model
from app.models.rag_chunk import RagChunk


CHUNKS_PATH = Path("data/processed/corpus_chunks.jsonl")

BATCH_SIZE = 32
EMBEDDING_DIMENSION = 1024


def load_chunks(path: Path) -> list[dict]:
    """Load chunk records from JSONL."""

    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def upsert_batch(
    session: Session,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Insert or update one embedding batch."""

    records = []

    for chunk, embedding in zip(chunks, embeddings):
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION} dimensions, "
                f"got {len(embedding)} for {chunk['chunk_id']}"
            )

        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "parent_page_id": chunk["parent_page_id"],
                "domain": chunk["domain"],
                "document_id": chunk["document_id"],
                "source_file": chunk["source_file"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "chunking_strategy": chunk["chunking_strategy"],
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "has_visual": chunk["has_visual"],
                "ocr_candidate": chunk["ocr_candidate"],
                "rendered_page": chunk.get("rendered_page"),
                "embedding": embedding,
            }
        )

    statement = insert(RagChunk).values(records)

    statement = statement.on_conflict_do_update(
        index_elements=[RagChunk.chunk_id],
        set_={
            "embedding": statement.excluded.embedding,
            "text": statement.excluded.text,
            "token_count": statement.excluded.token_count,
        },
    )

    session.execute(statement)
    session.commit()


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_PATH}"
        )

    chunks = load_chunks(CHUNKS_PATH)

    print(f"Loaded chunks: {len(chunks)}")

    embedding_model = create_embedding_model()

    engine = create_engine(settings.DATABASE_URL)

    total = len(chunks)

    with Session(engine) as session:
        for start in range(0, total, BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]

            texts = [
                chunk["text"]
                for chunk in batch
            ]

            embeddings = embedding_model.embed_documents(texts)

            upsert_batch(
                session=session,
                chunks=batch,
                embeddings=embeddings,
            )

            end = min(start + BATCH_SIZE, total)

            print(
                f"Embedded {end}/{total} chunks"
            )

    print()
    print("=" * 60)
    print("BGE-M3 EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"Chunks indexed: {total}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")


if __name__ == "__main__":
    main()