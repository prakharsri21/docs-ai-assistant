from __future__ import annotations

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from pgvector.sqlalchemy import Vector

from app.models.base import Base


class RagChunk(Base):
    __tablename__ = "rag_chunks"

    chunk_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )

    parent_page_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    domain: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    document_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    source_file: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    page: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    chunking_strategy: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    token_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    has_visual: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    ocr_candidate: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    rendered_page: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(1024),
        nullable=False,
    )