from __future__ import annotations


def validate_citations(
    answer: dict,
    retrieved_results: list[dict],
) -> dict:
    """
    Validate that every citation in an answer points to a
    document actually retrieved for that pipeline.
    """

    citations = answer.get("citations", [])

    if not citations:
        return {
            "valid": False,
            "errors": [
                "Answer contains no citations."
            ],
            "validated_citations": [],
        }

    retrieved_by_chunk = {
        result.get("chunk_id"): result
        for result in retrieved_results
        if result.get("chunk_id")
    }

    errors = []
    validated = []

    for citation in citations:

        chunk_id = citation.get("chunk_id")
        source_file = citation.get("source_file")
        page = citation.get("page")

        if not chunk_id:
            errors.append(
                "Citation is missing chunk_id."
            )
            continue

        retrieved = retrieved_by_chunk.get(
            chunk_id
        )

        if retrieved is None:
            errors.append(
                f"Citation references chunk "
                f"'{chunk_id}' which was not retrieved."
            )
            continue

        if source_file != retrieved.get(
            "source_file"
        ):
            errors.append(
                f"Source mismatch for {chunk_id}: "
                f"citation={source_file}, "
                f"retrieved={retrieved.get('source_file')}"
            )
            continue

        if page != retrieved.get("page"):
            errors.append(
                f"Page mismatch for {chunk_id}: "
                f"citation={page}, "
                f"retrieved={retrieved.get('page')}"
            )
            continue

        validated.append(
            {
                "chunk_id": chunk_id,
                "source_file": source_file,
                "page": page,
            }
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "validated_citations": validated,
    }