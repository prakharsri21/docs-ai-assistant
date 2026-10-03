from __future__ import annotations

import json

from app.graph.workflow import graph


QUERY = (
    "What happens when the batch size is one "
    "in stochastic gradient descent?"
)


if __name__ == "__main__":
    print("=" * 80)
    print("DAY 12 — FULL RAG GRAPH TEST")
    print("=" * 80)

    print()
    print("Query:")
    print(QUERY)

    result = graph.invoke({"query": QUERY})
    response = result["final_response"]

    print()
    print("=" * 80)
    print("CITATION VALIDATION")
    print("=" * 80)

    for name in (
        "basic_rag",
        "hybrid_rag",
        "okf_hybrid_rag",
    ):
        validation = response[name]["citation_validation"]

        print()
        print(name)
        print("Valid:", validation["valid"])

        if validation["errors"]:
            print("Errors:")
            for error in validation["errors"]:
                print(" -", error)

    print()
    print("=" * 80)
    print("BASIC RAG")
    print("=" * 80)
    print(response["basic_rag"]["answer"])

    print()
    print("Citations:")
    for citation in response["basic_rag"]["citations"]:
        print(
            f"- {citation['source_file']} "
            f"page {citation['page']} "
            f"({citation['chunk_id']})"
        )

    print()
    print("=" * 80)
    print("HYBRID RAG")
    print("=" * 80)
    print(response["hybrid_rag"]["answer"])

    print()
    print("Citations:")
    for citation in response["hybrid_rag"]["citations"]:
        print(
            f"- {citation['source_file']} "
            f"page {citation['page']} "
            f"({citation['chunk_id']})"
        )

    print()
    print("=" * 80)
    print("OKF + HYBRID RAG")
    print("=" * 80)
    print(response["okf_hybrid_rag"]["answer"])

    print()
    print("Citations:")
    for citation in response["okf_hybrid_rag"]["citations"]:
        print(
            f"- {citation['source_file']} "
            f"page {citation['page']} "
            f"({citation['chunk_id']})"
        )

    print()
    print("=" * 80)
    print("RAW RESPONSE")
    print("=" * 80)
    print(json.dumps(response, indent=2))
