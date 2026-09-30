from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.retrieval.dense_retriever import DenseRetriever
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class Source:
    source_file: str
    page: int
    chunk_id: str
    similarity: float


class BasicRAG:
    """
    Basic Retrieval-Augmented Generation pipeline.

    Flow:

        Question
            ↓
        Dense retrieval
            ↓
        Context construction
            ↓
        LLM generation
            ↓
        Answer + sources
    """

    def __init__(self) -> None:
        self.retriever = DenseRetriever()

        model_name = os.getenv(
            "OPENAI_MODEL",
            "gpt-5.6-luna",
        )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a question-answering assistant for an MIT
Machine Learning and Deep Learning course corpus.

Use ONLY the retrieved context to answer the question.

Rules:
1. Do not invent facts that are not supported by the context.
2. If the context is insufficient, say:
   "The retrieved course material does not provide
   enough information to answer this confidently."
3. Prefer direct evidence from the retrieved material.
4. Keep the answer clear and concise.
5. Cite the source using the source labels provided
   in the context.

Retrieved context:

{context}
""",
                ),
                (
                    "human",
                    "{question}",
                ),
            ]
        )

        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def build_context(
        self,
        results: list[dict],
    ) -> tuple[str, list[Source]]:
        """
        Convert retrieved database rows into LLM context.
        """

        context_parts = []
        sources = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            source_label = (
                f"[{index}] "
                f"{result['source_file']}, "
                f"page {result['page']}"
            )

            context_parts.append(
                "\n".join(
                    [
                        source_label,
                        result["text"],
                    ]
                )
            )

            sources.append(
                Source(
                    source_file=result["source_file"],
                    page=result["page"],
                    chunk_id=result["chunk_id"],
                    similarity=float(
                        result["similarity"]
                    ),
                )
            )

        context = "\n\n".join(
            context_parts
        )

        return context, sources

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:
        """
        Run the complete Basic RAG pipeline.
        """

        results = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        if not results:
            return {
                "answer": (
                    "No relevant course material "
                    "was retrieved."
                ),
                "sources": [],
            }

        context, sources = self.build_context(
            results
        )

        answer = self.chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "source_file": source.source_file,
                    "page": source.page,
                    "chunk_id": source.chunk_id,
                    "similarity": source.similarity,
                }
                for source in sources
            ],
        }