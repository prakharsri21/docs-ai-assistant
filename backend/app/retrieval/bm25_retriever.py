from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


CHUNKS_PATH = Path(
    "data/processed/corpus_chunks.jsonl"
)


def tokenize(text: str) -> list[str]:
    """
    Normalize technical terminology before BM25 indexing/search.

    The same preprocessing is applied to both documents and queries.
    """

    text = text.lower()

    # Normalize common technical phrases.
    text = re.sub(
        r"\bbatch[\s_-]+size\b",
        "batchsize",
        text,
    )

    # Normalize a few simple number words while preserving digits.
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    for word, digit in number_words.items():
        text = re.sub(
            rf"\b{word}\b",
            digit,
            text,
        )

    return re.findall(
        r"[A-Za-z0-9_]+",
        text,
    )


class BM25Retriever:
    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
    ) -> None:

        self.chunks = self._load_chunks(
            chunks_path
        )

        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    @staticmethod
    def _load_chunks(
        path: Path,
    ) -> list[dict]:

        chunks = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                if line.strip():
                    chunks.append(
                        json.loads(line)
                    )

        return chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:

        query_tokens = tokenize(query)

        ranked_indexes = (
            self.bm25
            .get_top_n(
                query_tokens,
                self.chunks,
                n=top_k,
            )
        )

        results = []

        for rank, chunk in enumerate(
            ranked_indexes,
            start=1,
        ):
            results.append(
                {
                    **chunk,
                    "bm25_rank": rank,
                }
            )

        return results

def main() -> None:

    retriever = BM25Retriever()

    query = (
        "What happens when the batch "
        "contains only one training example?"
    )

    results = retriever.retrieve(
        query,
        top_k=10,
    )

    print()
    print("=" * 80)
    print("BM25 RETRIEVAL TEST")
    print("=" * 80)
    print(f"Query: {query}")
    print()

    for result in results:
        print(
            f"{result['bm25_rank']:2}. "
            f"{result['source_file']} | "
            f"p.{result['page']} | "
            f"{result['chunk_id']}"
        )

        print(
            result["text"][:300]
            .replace("\n", " ")
        )

        print()


if __name__ == "__main__":
    main()