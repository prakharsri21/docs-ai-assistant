from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


KnowledgeType = Literal[
    "concept",
    "definition",
    "method",
    "algorithm",
    "equation",
    "example",
    "theorem",
    "architecture",
    "principle",
    "property",
    "process",
]


class OKFSource(BaseModel):
    document_id: str
    source_file: str
    page: int
    chunk_id: str


class OKFRelationship(BaseModel):
    type: str
    target: str


class OKFNode(BaseModel):
    id: str
    type: KnowledgeType
    domain: str
    title: str

    source: OKFSource

    summary: str = ""
    content: str = ""

    relationships: list[OKFRelationship] = Field(default_factory=list)

    keywords: list[str] = Field(default_factory=list)