from __future__ import annotations

import json
from pathlib import Path


INPUT_PATH = Path("data/okf/okf_nodes.jsonl")
USABLE_PATH = Path("data/okf/okf_nodes_usable.jsonl")
QUARANTINE_PATH = Path("data/okf/okf_quarantine.jsonl")


PLACEHOLDER_PHRASES = (
    "no content available",
    "uninterpretable content",
    "uninterpretable mathematical expression",
    "page number reference",
    "unspecified content",
)


def is_usable(node: dict) -> tuple[bool, str | None]:
    title = (node.get("title") or "").strip().lower()
    summary = (node.get("summary") or "").strip()
    content = (node.get("content") or "").strip()
    keywords = node.get("keywords") or []

    # Missing core fields
    if not title:
        return False, "empty_title"

    if not summary:
        return False, "empty_summary"

    if not content:
        return False, "empty_content"

    # Placeholder / unusable LLM interpretation
    for phrase in PLACEHOLDER_PHRASES:
        if phrase in title:
            return False, f"placeholder_title:{phrase}"

    # Extremely weak content
    if len(content) < 40:
        return False, "content_too_short"

    # No retrieval metadata at all
    if not keywords and len(summary) < 50:
        return False, "insufficient_retrieval_metadata"

    return True, None


def main():
    usable = 0
    quarantined = 0

    USABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with (
        INPUT_PATH.open("r", encoding="utf-8") as source,
        USABLE_PATH.open("w", encoding="utf-8") as good,
        QUARANTINE_PATH.open("w", encoding="utf-8") as bad,
    ):
        for line in source:
            if not line.strip():
                continue

            node = json.loads(line)

            valid, reason = is_usable(node)

            if valid:
                good.write(
                    json.dumps(
                        node,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                usable += 1

            else:
                record = {
                    "reason": reason,
                    "node": node,
                }

                bad.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                quarantined += 1

    print("=" * 60)
    print("OKF QUALITY AUDIT")
    print("=" * 60)
    print("Usable nodes     :", usable)
    print("Quarantined nodes:", quarantined)
    print("Total            :", usable + quarantined)
    print()
    print("Usable output    :", USABLE_PATH)
    print("Quarantine       :", QUARANTINE_PATH)


if __name__ == "__main__":
    main()