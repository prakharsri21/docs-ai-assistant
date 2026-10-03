from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerCitation(BaseModel):
    chunk_id: str
    source_file: str
    page: int


class RAGAnswer(BaseModel):
    answer: str
    citations: list[AnswerCitation] = Field(
        default_factory=list
    )