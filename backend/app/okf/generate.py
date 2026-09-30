from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.okf.extractor import OKFExtractor


INPUT_PATH = Path("data/processed/corpus_chunks.jsonl")
OUTPUT_PATH = Path("data/okf/okf_nodes.jsonl")
FAILURES_PATH = Path("data/okf/okf_failures.jsonl")


def load_completed_ids() -> set[str]:
    """Load IDs already successfully processed."""
    completed = set()

    if not OUTPUT_PATH.exists():
        return completed

    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            node_id = record.get("id")

            if node_id:
                completed.add(node_id)

    return completed


def load_chunks(limit: int | None = None):
    """Load source chunks from the corpus."""
    chunks = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            chunks.append(json.loads(line))

            if limit is not None and len(chunks) >= limit:
                break

    return chunks


def append_jsonl(path: Path, record: dict):
    """Append one JSON record and flush immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def process_chunk(
    extractor: OKFExtractor,
    chunk: dict,
    max_retries: int = 3,
):
    """Extract one OKF node with retry handling."""

    last_error = None

    for attempt in range(1, max_retries + 1):

        try:
            node = extractor.extract(chunk)

            # Make identity deterministic.
            node.id = chunk["chunk_id"]

            return node

        except Exception as exc:
            last_error = exc

            wait_seconds = 2 ** (attempt - 1)

            print(
                f"  Attempt {attempt}/{max_retries} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < max_retries:
                print(
                    f"  Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)

    raise last_error


def main():
    parser = argparse.ArgumentParser(
        description="Generate OKF nodes from corpus chunks."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N chunks.",
    )

    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = load_chunks(args.limit)

    completed_ids = load_completed_ids()

    extractor = OKFExtractor()

    total = len(chunks)
    skipped = 0
    successful = 0
    failed = 0

    print("=" * 60)
    print("OKF GENERATION")
    print("=" * 60)
    print(f"Input chunks : {total}")
    print(f"Already done : {len(completed_ids)}")
    print()

    for index, chunk in enumerate(chunks, start=1):

        chunk_id = chunk["chunk_id"]

        if chunk_id in completed_ids:
            skipped += 1
            print(
                f"[{index}/{total}] SKIP {chunk_id}"
            )
            continue

        print(
            f"[{index}/{total}] Processing {chunk_id}"
        )

        try:
            node = process_chunk(
                extractor,
                chunk,
            )

            append_jsonl(
                OUTPUT_PATH,
                node.model_dump(mode="json"),
            )

            completed_ids.add(chunk_id)

            successful += 1

            print(
                f"  ✓ {node.title}"
            )

        except Exception as exc:
            failed += 1

            failure_record = {
                "chunk_id": chunk_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

            append_jsonl(
                FAILURES_PATH,
                failure_record,
            )

            print(
                f"  ✗ FAILED: {type(exc).__name__}: {exc}"
            )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Successful : {successful}")
    print(f"Skipped    : {skipped}")
    print(f"Failed     : {failed}")
    print()
    print(f"Output     : {OUTPUT_PATH}")
    print(f"Failures   : {FAILURES_PATH}")


if __name__ == "__main__":
    main()
    