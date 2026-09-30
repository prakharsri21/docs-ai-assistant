from __future__ import annotations

from collections import defaultdict


DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: dict[str, list[dict]],
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """
    Combine multiple ranked retrieval lists using RRF.

    Parameters
    ----------
    result_lists:
        Mapping such as:
        {
            "dense": [...],
            "bm25": [...],
        }

    rrf_k:
        RRF smoothing constant.
    """

    scores = defaultdict(float)
    documents = {}
    ranks = defaultdict(dict)

    for retriever_name, results in result_lists.items():

        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            scores[chunk_id] += (
                1.0 / (rrf_k + rank)
            )

            documents[chunk_id] = result

            ranks[chunk_id][retriever_name] = rank

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    output = []

    for chunk_id, score in ranked:

        result = dict(
            documents[chunk_id]
        )

        result["rrf_score"] = score
        result["retriever_ranks"] = ranks[chunk_id]
        result["retrieved_by"] = list(
            ranks[chunk_id].keys()
        )

        output.append(result)

    return output