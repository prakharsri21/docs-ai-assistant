from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):

    query: str

    basic_results: list[dict[str, Any]]
    hybrid_results: list[dict[str, Any]]
    okf_results: list[dict[str, Any]]

    basic_answer: dict[str, Any]
    hybrid_answer: dict[str, Any]
    okf_answer: dict[str, Any]

    validation: dict[str, dict[str, Any]]

    final_response: dict[str, Any]

    errors: list[str]