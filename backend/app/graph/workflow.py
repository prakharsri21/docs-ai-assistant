from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    basic_answer_node,
    basic_retrieval_node,
    finalize_node,
    hybrid_answer_node,
    hybrid_retrieval_node,
    okf_answer_node,
    okf_retrieval_node,
)
from app.graph.state import RAGState


def build_graph():
    builder = StateGraph(RAGState)

    # Retrieval nodes
    builder.add_node(
        "basic_retrieval",
        basic_retrieval_node,
    )

    builder.add_node(
        "hybrid_retrieval",
        hybrid_retrieval_node,
    )

    builder.add_node(
        "okf_retrieval",
        okf_retrieval_node,
    )

    # Answer nodes
    builder.add_node(
        "basic_answer",
        basic_answer_node,
    )

    builder.add_node(
        "hybrid_answer",
        hybrid_answer_node,
    )

    builder.add_node(
        "okf_answer",
        okf_answer_node,
    )

    # Final assembly
    builder.add_node(
        "finalize",
        finalize_node,
    )

    # Fan-out
    builder.add_edge(START, "basic_retrieval")
    builder.add_edge(START, "hybrid_retrieval")
    builder.add_edge(START, "okf_retrieval")

    # Retrieval → answer
    builder.add_edge(
        "basic_retrieval",
        "basic_answer",
    )

    builder.add_edge(
        "hybrid_retrieval",
        "hybrid_answer",
    )

    builder.add_edge(
        "okf_retrieval",
        "okf_answer",
    )

    # Fan-in
    builder.add_edge(
        [
            "basic_answer",
            "hybrid_answer",
            "okf_answer",
        ],
        "finalize",
    )

    builder.add_edge("finalize", END)

    return builder.compile()


graph = build_graph()