from __future__ import annotations

import json
import re
from pathlib import Path

from app.embeddings.bge_m3 import create_embedding_model
from app.retrieval.rrf import reciprocal_rank_fusion


OKF_PATH = Path("data/okf/okf_nodes_usable.jsonl")


def tokenize(text: str) -> list[str]:
    """
    Lightweight tokenizer for BM25.

    Normalize common variants so phrases like:
    'batch-size', 'batch size', and 'batch_size'
    are more likely to match.
    """
    text = text.lower()

    text = re.sub(
        r"\bbatch[\s_-]+size\b",
        "batchsize",
        text,
    )

    return re.findall(r"[a-z0-9_]+", text)


class OKFRetriever:
    def __init__(
        self,
        okf_path: Path = OKF_PATH,
    ):
        self.okf_path = okf_path

        self.nodes = self._load_nodes()

        self.embedding_model = create_embedding_model()

        # Build searchable representations once.
        self.search_texts = [
            self._build_search_text(node)
            for node in self.nodes
        ]

        self.embeddings = self.embedding_model.embed_documents(
            self.search_texts
        )

        self.bm25_corpus = [
            tokenize(text)
            for text in self.search_texts
        ]

        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(self.bm25_corpus)

    def _load_nodes(self) -> list[dict]:
        if not self.okf_path.exists():
            raise FileNotFoundError(
                f"OKF file not found: {self.okf_path}"
            )

        nodes = []

        with self.okf_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if line.strip():
                    nodes.append(json.loads(line))

        if not nodes:
            raise ValueError(
                "No usable OKF nodes were found."
            )

        return nodes

    @staticmethod
    def _build_search_text(node: dict) -> str:
        relationships = node.get(
            "relationships",
            [],
        )

        relationship_text = " ".join(
            f"{r.get('type', '')} {r.get('target', '')}"
            for r in relationships
        )

        keywords = " ".join(
            node.get("keywords", [])
        )

        return "\n".join(
            [
                f"Title: {node.get('title', '')}",
                f"Type: {node.get('type', '')}",
                f"Summary: {node.get('summary', '')}",
                f"Keywords: {keywords}",
                f"Relationships: {relationship_text}",
                f"Content: {node.get('content', '')}",
            ]
        )

    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:

        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        # Cosine similarity because the BGE-M3 wrapper
        # normalizes embeddings.
        scores = []

        for index, embedding in enumerate(
            self.embeddings
        ):
            similarity = sum(
                a * b
                for a, b in zip(
                    query_embedding,
                    embedding,
                )
            )

            scores.append(
                (index, similarity)
            )

        scores.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for rank, (index, score) in enumerate(
            scores[:top_k],
            start=1,
        ):
            node = self.nodes[index]

            results.append(
                {
                    "okf_id": node["id"],
                    "chunk_id": node["source"]["chunk_id"],
                    "source_file": node["source"]["source_file"],
                    "page": node["source"]["page"],
                    "title": node["title"],
                    "summary": node["summary"],
                    "content": node["content"],
                    "keywords": node["keywords"],
                    "dense_score": float(score),
                    "dense_rank": rank,
                }
            )

        return results

    def _bm25_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:

        tokens = tokenize(query)

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1,
        ):
            node = self.nodes[index]

            results.append(
                {
                    "okf_id": node["id"],
                    "chunk_id": node["source"]["chunk_id"],
                    "source_file": node["source"]["source_file"],
                    "page": node["source"]["page"],
                    "title": node["title"],
                    "summary": node["summary"],
                    "content": node["content"],
                    "keywords": node["keywords"],
                    "bm25_score": float(scores[index]),
                    "bm25_rank": rank,
                }
            )

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
    ) -> list[dict]:

        dense_results = self._dense_retrieve(
            query,
            candidate_k,
        )

        bm25_results = self._bm25_retrieve(
            query,
            candidate_k,
        )

        fused = reciprocal_rank_fusion(
            {
                "okf_dense": dense_results,
                "okf_bm25": bm25_results,
            }
        )

        return fused[:top_k]
        