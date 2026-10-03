from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from app.graph.validation import validate_citations
from app.graph.state import RAGState

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.reranked_hybrid_retriever import (
    RerankedHybridRetriever,
)
from app.retrieval.reranked_okf_retriever import (
    RerankedOKFRetriever,
)
from app.schemas.answer import RAGAnswer


load_dotenv(override=True)


# -------------------------------------------------------------------
# Shared components
# -------------------------------------------------------------------

dense_retriever = DenseRetriever()
hybrid_retriever = RerankedHybridRetriever()
okf_retriever = RerankedOKFRetriever()


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant answering questions about the supplied
machine learning and deep learning course material.

Use ONLY the supplied retrieved context.

Rules:

1. Do not invent facts.
2. Do not use knowledge outside the retrieved context.
3. If the context is insufficient, explicitly say so.
4. Give a clear, concise explanation.
5. Citations must refer ONLY to sources provided in the context.
6. Never invent a chunk_id, filename, or page number.
7. Preserve equations and technical terminology when relevant.

Return a structured answer containing:

- answer
- citations

Each citation must contain:

- chunk_id
- source_file
- page

Retrieved context:

{context}
""",
        ),
        (
            "human",
            "{query}",
        ),
    ]
)


# -------------------------------------------------------------------
# Context formatting
# -------------------------------------------------------------------

def build_context(
    results: list[dict],
    max_results: int = 8,
) -> str:
    sections = []

    for rank, result in enumerate(
        results[:max_results],
        start=1,
    ):
        chunk_id = result.get(
            "chunk_id",
            result.get("okf_id", ""),
        )

        source_file = result.get(
            "source_file",
            "",
        )

        page = result.get(
            "page",
            "",
        )

        title = result.get(
            "title",
            "",
        )

        text = result.get(
            "text",
            "",
        )

        if not text:
            text = result.get(
                "content",
                "",
            )

        summary = result.get(
            "summary",
            "",
        )

        sections.append(
            f"""
SOURCE #{rank}

chunk_id: {chunk_id}

source_file: {source_file}

page: {page}

title: {title}

summary:

{summary} 

content:

{text}
""".strip()
        )

    return "\n\n---\n\n".join(sections)


# -------------------------------------------------------------------
# Answer generation
# -------------------------------------------------------------------

def generate_answer(
    query: str,
    results: list[dict],
) -> dict:

    start = time.perf_counter()

    context = build_context(results)

    messages = ANSWER_PROMPT.format_messages(
        query=query,
        context=context,
    )

    structured_llm = llm.with_structured_output(
        RAGAnswer
    )

    response = structured_llm.invoke(
        messages
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"[LATENCY] answer_generation: "
        f"{elapsed_ms:.0f} ms"
    )

    return response.model_dump()


# -------------------------------------------------------------------
# Retrieval nodes
# -------------------------------------------------------------------

def basic_retrieval_node(
    state: RAGState,
) -> dict:
    start = time.perf_counter()

    results = dense_retriever.retrieve(
        state["query"],
        top_k=10,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"[LATENCY] basic_retrieval: "
        f"{elapsed_ms:.0f} ms"
    )

    return {
        "basic_results": results,
    }


def hybrid_retrieval_node(
    state: RAGState,
) -> dict:
    start = time.perf_counter()

    results = hybrid_retriever.retrieve(
        state["query"],
        top_k=10,
    )

    print(
        f"[LATENCY] hybrid_retrieval: "
        f"{(time.perf_counter() - start) * 1000:.0f} ms"
    )

    return {
        "hybrid_results": results,
    }


def okf_retrieval_node(
    state: RAGState,
) -> dict:
    start = time.perf_counter()

    results = okf_retriever.retrieve(
        state["query"],
        top_k=10,
    )

    print(
        f"[LATENCY] okf_retrieval: "
        f"{(time.perf_counter() - start) * 1000:.0f} ms"
    )

    return {
        "okf_results": results,
    }


# -------------------------------------------------------------------
# Answer nodes
# -------------------------------------------------------------------

def basic_answer_node(
    state: RAGState,
) -> dict:
    answer = generate_answer(
        state["query"],
        state["basic_results"],
    )

    return {
        "basic_answer": answer,
    }


def hybrid_answer_node(
    state: RAGState,
) -> dict:
    answer = generate_answer(
        state["query"],
        state["hybrid_results"],
    )

    return {
        "hybrid_answer": answer,
    }


def okf_answer_node(
    state: RAGState,
) -> dict:
    answer = generate_answer(
        state["query"],
        state["okf_results"],
    )

    return {
        "okf_answer": answer,
    }


# -------------------------------------------------------------------
# Final response
# -------------------------------------------------------------------

def finalize_node(
    state: RAGState,
) -> dict:
    basic_validation = validate_citations(
        state["basic_answer"],
        state["basic_results"],
    )

    hybrid_validation = validate_citations(
        state["hybrid_answer"],
        state["hybrid_results"],
    )

    okf_validation = validate_citations(
        state["okf_answer"],
        state["okf_results"],
    )

    validation = {
        "basic_rag": basic_validation,
        "hybrid_rag": hybrid_validation,
        "okf_hybrid_rag": okf_validation,
    }

    return {
        "validation": validation,
        "final_response": {
            "query": state["query"],
            "basic_rag": {
                **state["basic_answer"],
                "citation_validation": basic_validation,
            },
            "hybrid_rag": {
                **state["hybrid_answer"],
                "citation_validation": hybrid_validation,
            },
            "okf_hybrid_rag": {
                **state["okf_answer"],
                "citation_validation": okf_validation,
            },
        },
    }