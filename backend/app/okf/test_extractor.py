import json

from app.okf.extractor import OKFExtractor


CHUNKS_PATH = "data/processed/corpus_chunks.jsonl"
TARGET_CHUNK_ID = "deep_learning_lec2-p9-c01"


def load_target_chunk():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)

            if chunk["chunk_id"] == TARGET_CHUNK_ID:
                return chunk

    raise ValueError(
        f"Chunk not found: {TARGET_CHUNK_ID}"
    )


if __name__ == "__main__":

    chunk = load_target_chunk()

    print("Testing chunk:")
    print(chunk["chunk_id"])
    print(f"Source: {chunk['source_file']}, page {chunk['page']}")
    print()
    print(chunk["text"])
    print()

    extractor = OKFExtractor()

    node = extractor.extract(chunk)

    print("\n=== OKF NODE ===")
    print(node.model_dump_json(indent=2))