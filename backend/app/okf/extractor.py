from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.okf.prompts import OKF_EXTRACTION_PROMPT
from app.okf.schema import OKFNode

load_dotenv(override=True)


class OKFExtractor:

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def extract(self, chunk: dict) -> OKFNode:

        prompt = OKF_EXTRACTION_PROMPT.format(
            chunk_text=chunk["text"],
            document_id=chunk["document_id"],
            source_file=chunk["source_file"],
            page=chunk["page"],
            chunk_id=chunk["chunk_id"],
            domain=chunk["domain"],
        )

        structured_llm = self.llm.with_structured_output(OKFNode)

        result = structured_llm.invoke(prompt)

        try:
            node = OKFNode.model_validate(result)

            # Preserve deterministic identity and provenance.
            # The source chunk is the canonical identity of this OKF node.
            node.id = chunk["chunk_id"]

            return node

        except ValidationError as exc:
            raise ValueError(
                f"Invalid OKF output for {chunk['chunk_id']}"
            ) from exc