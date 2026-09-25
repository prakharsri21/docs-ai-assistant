from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TOKENIZER_NAME = "BAAI/bge-m3"

TARGET_TOKENS = 450
MAX_TOKENS = 650
SHORT_PAGE_THRESHOLD = 600

# Used when we need overlap between logical blocks.
MAX_OVERLAP_TOKENS = 80


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME
)


def token_count(text: str) -> int:
    """Return the number of model tokens in a text."""
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


# ---------------------------------------------------------
# Data model
# ---------------------------------------------------------

@dataclass
class ChunkRecord:
    chunk_id: str
    parent_page_id: str

    domain: str
    document_id: str
    source_file: str
    page: int

    chunk_index: int
    chunking_strategy: str

    text: str
    token_count: int

    has_visual: bool
    ocr_candidate: bool
    rendered_page: str | None


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace without destroying line structure."""
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    # Remove repeated blank lines.
    result: list[str] = []
    previous_blank = False

    for line in lines:
        if not line:
            if not previous_blank:
                result.append("")

            previous_blank = True
        else:
            result.append(line)
            previous_blank = False

    return "\n".join(result).strip()


def looks_like_heading(line: str) -> bool:
    """
    Heuristic heading detection.

    This is intentionally conservative because PDF text extraction
    does not always preserve font/style information.
    """

    line = line.strip()

    if not line:
        return False

    if len(line) > 100:
        return False

    # Markdown-like headings if they somehow survived extraction.
    if re.match(r"^#{1,6}\s+", line):
        return True

    # Numbered headings:
    # 1.
    # 1.2
    # 2.3.1
    if re.match(r"^\d+(?:\.\d+)*[\.)]?\s+\S+", line):
        return True

    # Common lecture heading patterns.
    if re.match(
        r"^(Lecture|Topic|Overview|Summary|Introduction|"
        r"Conclusion|Example|Definition|Theorem|Proof)\b",
        line,
        flags=re.IGNORECASE,
    ):
        return True

    # Very short all-caps lines are often slide headings.
    letters = [c for c in line if c.isalpha()]

    if (
        len(letters) >= 4
        and line.upper() == line
        and not line.endswith(".")
        and len(line.split()) <= 12
    ):
        return True

    return False


def is_bullet(line: str) -> bool:
    """Detect common bullet/list prefixes."""
    return bool(
        re.match(
            r"^(?:[-•▪◦*]|\d+[.)])\s+",
            line.strip(),
        )
    )


def build_blocks(text: str) -> list[str]:
    """
    Turn page text into soft logical blocks.

    This does not pretend to perfectly reconstruct PDF layout.
    It creates useful boundaries while preserving the original
    page-level text.
    """

    lines = normalize_whitespace(text).splitlines()

    if not lines:
        return []

    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current

        if current:
            block = "\n".join(current).strip()

            if block:
                blocks.append(block)

        current = []

    for index, line in enumerate(lines):

        if not line:
            flush()
            continue

        heading = looks_like_heading(line)

        # Start a new block at a heading.
        if heading and current:
            flush()

        current.append(line)

        # If this is a heading, keep it separate from the following body.
        if heading:
            flush()
            continue

        # Bullet sequences are kept together unless there is a clear
        # transition.
        if is_bullet(line):
            continue

        # A sentence-ending line followed by a probable new short
        # section is treated as a soft boundary.
        if (
            index + 1 < len(lines)
            and lines[index + 1]
            and looks_like_heading(lines[index + 1])
        ):
            flush()

    flush()

    return blocks


# ---------------------------------------------------------
# LangChain fallback splitter
# ---------------------------------------------------------

def split_oversized_block(
    block: str,
    metadata: dict[str, Any],
) -> list[Document]:
    """
    Use LangChain only when one logical block is too large
    to fit into our retrieval window.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_TOKENS,
        chunk_overlap=60,
        length_function=token_count,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
        keep_separator=True,
    )

    document = Document(
        page_content=block,
        metadata=metadata,
    )

    return splitter.split_documents(
        [document]
    )


# ---------------------------------------------------------
# Chunk one page
# ---------------------------------------------------------

def chunk_page(
    page: dict[str, Any],
) -> list[ChunkRecord]:

    page_id = (
        f"{page['document_id']}"
        f"-p{page['page']}"
    )

    text = normalize_whitespace(
        page.get("text", "")
    )

    if not text:
        return []

    page_tokens = token_count(text)

    metadata = {
        "domain": page["domain"],
        "document_id": page["document_id"],
        "source_file": page["source_file"],
        "page": page["page"],
        "parent_page_id": page_id,
    }

    # -----------------------------------------------------
    # Rule 1:
    # Short pages remain intact.
    # This is especially useful for lecture slides.
    # -----------------------------------------------------

    if page_tokens <= SHORT_PAGE_THRESHOLD:

        return [
            ChunkRecord(
                chunk_id=f"{page_id}-c01",
                parent_page_id=page_id,
                domain=page["domain"],
                document_id=page["document_id"],
                source_file=page["source_file"],
                page=page["page"],
                chunk_index=1,
                chunking_strategy="page_preserving",
                text=text,
                token_count=page_tokens,
                has_visual=page.get(
                    "visual_candidate",
                    False,
                ),
                ocr_candidate=page.get(
                    "ocr_candidate",
                    False,
                ),
                rendered_page=page.get(
                    "rendered_page"
                ),
            )
        ]

    # -----------------------------------------------------
    # Rule 2:
    # Long page → logical blocks.
    # -----------------------------------------------------

    blocks = build_blocks(text)

    # Fallback if block detection failed.
    if not blocks:
        blocks = [text]

    final_texts: list[str] = []

    current_blocks: list[str] = []
    current_tokens = 0

    for block in blocks:

        block_tokens = token_count(block)

        # Very large logical block.
        if block_tokens > MAX_TOKENS:

            if current_blocks:
                final_texts.append(
                    "\n\n".join(current_blocks)
                )

                current_blocks = []
                current_tokens = 0

            sub_docs = split_oversized_block(
                block,
                metadata,
            )

            final_texts.extend(
                doc.page_content
                for doc in sub_docs
            )

            continue

        # Fits into current chunk.
        if (
            current_tokens + block_tokens
            <= TARGET_TOKENS
        ):

            current_blocks.append(block)
            current_tokens += block_tokens
            continue

        # Current chunk is full.
        if current_blocks:
            final_texts.append(
                "\n\n".join(current_blocks)
            )

        # Keep a small amount of logical overlap.
        overlap_blocks: list[str] = []

        if current_blocks:
            previous = current_blocks[-1]

            if token_count(previous) <= MAX_OVERLAP_TOKENS:
                overlap_blocks = [previous]

        current_blocks = overlap_blocks + [block]

        current_tokens = sum(
            token_count(item)
            for item in current_blocks
        )

    if current_blocks:
        final_texts.append(
            "\n\n".join(current_blocks)
        )

    # -----------------------------------------------------
    # Build child records
    # -----------------------------------------------------

    chunks: list[ChunkRecord] = []

    for index, chunk_text in enumerate(
        final_texts,
        start=1,
    ):

        chunks.append(
            ChunkRecord(
                chunk_id=f"{page_id}-c{index:02d}",
                parent_page_id=page_id,
                domain=page["domain"],
                document_id=page["document_id"],
                source_file=page["source_file"],
                page=page["page"],
                chunk_index=index,
                chunking_strategy="hierarchical_structure_aware",
                text=chunk_text,
                token_count=token_count(chunk_text),
                has_visual=page.get(
                    "visual_candidate",
                    False,
                ),
                ocr_candidate=page.get(
                    "ocr_candidate",
                    False,
                ),
                rendered_page=page.get(
                    "rendered_page"
                ),
            )
        )

    return chunks


# ---------------------------------------------------------
# Read / write
# ---------------------------------------------------------

def read_pages(
    path: Path,
) -> list[dict[str, Any]]:

    pages = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            pages.append(
                json.loads(line)
            )

    return pages


def write_chunks(
    path: Path,
    chunks: list[ChunkRecord],
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in chunks:
            file.write(
                json.dumps(
                    asdict(chunk),
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def build_stats(
    chunks: list[ChunkRecord],
    page_count: int,
) -> dict[str, Any]:

    token_counts = [
        chunk.token_count
        for chunk in chunks
    ]

    visual_chunks = sum(
        chunk.has_visual
        for chunk in chunks
    )

    ocr_chunks = sum(
        chunk.ocr_candidate
        for chunk in chunks
    )

    return {
        "pages": page_count,
        "chunks": len(chunks),
        "visual_chunks": visual_chunks,
        "ocr_candidate_chunks": ocr_chunks,
        "avg_tokens": (
            sum(token_counts) / len(token_counts)
            if token_counts
            else 0
        ),
        "min_tokens": (
            min(token_counts)
            if token_counts
            else 0
        ),
        "max_tokens": (
            max(token_counts)
            if token_counts
            else 0
        ),
        "target_tokens": TARGET_TOKENS,
        "max_tokens": MAX_TOKENS,
        "short_page_threshold": SHORT_PAGE_THRESHOLD,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create hierarchical retrieval chunks "
            "from page-level corpus records."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/corpus_pages.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/corpus_chunks.jsonl"
        ),
    )

    parser.add_argument(
        "--stats",
        type=Path,
        default=Path(
            "data/processed/chunking_stats.json"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    pages = read_pages(args.input)

    chunks: list[ChunkRecord] = []

    for index, page in enumerate(
        pages,
        start=1,
    ):

        page_chunks = chunk_page(page)

        chunks.extend(page_chunks)

        if index % 100 == 0:
            print(
                f"Processed {index}/{len(pages)} pages"
            )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_chunks(
        args.output,
        chunks,
    )

    stats = build_stats(
        chunks,
        page_count=len(pages),
    )

    with args.stats.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            stats,
            file,
            indent=2,
        )

    print()
    print("=" * 55)
    print("CHUNKING COMPLETE")
    print("=" * 55)
    print(f"Pages:              {stats['pages']}")
    print(f"Chunks:             {stats['chunks']}")
    print(
        f"Visual chunks:      "
        f"{stats['visual_chunks']}"
    )
    print(
        f"OCR candidates:     "
        f"{stats['ocr_candidate_chunks']}"
    )
    print(
        f"Average tokens:     "
        f"{stats['avg_tokens']:.1f}"
    )
    print(
        f"Minimum tokens:     "
        f"{stats['min_tokens']}"
    )
    print(
        f"Maximum tokens:     "
        f"{stats['max_tokens']}"
    )
    print()
    print(f"Chunks: {args.output}")
    print(f"Stats:  {args.stats}")


if __name__ == "__main__":
    main()