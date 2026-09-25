from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf


@dataclass
class PageRecord:
    domain: str
    document_id: str
    source_file: str
    source_sha256: str

    page: int

    text: str
    char_count: int
    word_count: int

    image_count: int
    drawing_count: int

    full_page_image: bool
    visual_candidate: bool
    ocr_candidate: bool

    rendered_page: str | None


@dataclass
class DocumentRecord:
    domain: str
    document_id: str
    source_file: str
    source_sha256: str

    title: str | None

    page_count: int
    pages_with_text: int

    visual_pages: int
    ocr_candidates: int

    total_characters: int


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 for source-file provenance."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def clean_text(text: str) -> str:
    """
    Perform only light normalization.

    We deliberately avoid aggressive cleaning because
    chunking and layout analysis come later.
    """
    lines = [line.strip() for line in text.splitlines()]

    cleaned = []
    previous_blank = False

    for line in lines:
        if not line:
            if not previous_blank:
                cleaned.append("")

            previous_blank = True
            continue

        cleaned.append(line)
        previous_blank = False

    return "\n".join(cleaned).strip()


def render_page(
    page: pymupdf.Page,
    output_path: Path,
) -> None:
    """
    Render a page for visual inspection.

    We use a moderate resolution rather than producing
    unnecessarily large images.
    """
    matrix = pymupdf.Matrix(1.5, 1.5)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pixmap.save(str(output_path))


def analyze_page(
    page: pymupdf.Page,
) -> tuple[
    str,
    int,
    int,
    int,
    bool,
    bool,
    bool,
]:
    """
    Extract text and identify visual/OCR candidates.
    """

    text = clean_text(
        page.get_text(
            "text",
            sort=True,
        )
    )

    char_count = len(text)
    word_count = len(text.split())

    # Embedded/displayed image information.
    image_info = page.get_image_info(
        hashes=False,
        xrefs=False,
    )

    image_count = len(image_info)

    # Vector graphics / line-art / drawings.
    drawings = page.get_drawings()
    drawing_count = len(drawings)

    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    full_page_image = False

    if page_area > 0:
        for image in image_info:
            bbox = image.get("bbox")

            if bbox is None:
                continue

            image_area = (
                bbox.width * bbox.height
                if hasattr(bbox, "width")
                else 0
            )

            if image_area / page_area >= 0.80:
                full_page_image = True
                break

    # A page containing an image or a meaningful number
    # of vector paths is treated as a visual candidate.
    #
    # 20 is intentionally only a heuristic. We will
    # inspect the actual corpus and adjust it.
    visual_candidate = (
        image_count > 0
        or drawing_count >= 20
    )

    # We don't OCR everything.
    # Low-text pages containing visual content are
    # candidates for later OCR review.
    low_text = char_count < 100

    ocr_candidate = (
        low_text
        and (
            image_count > 0
            or drawing_count > 0
        )
    )

    return (
        text,
        char_count,
        word_count,
        image_count,
        drawing_count,
        full_page_image,
        visual_candidate,
        ocr_candidate,
    )


def extract_document(
    pdf_path: Path,
    domain: str,
    visual_root: Path,
) -> tuple[
    list[PageRecord],
    DocumentRecord,
]:
    """Extract one complete PDF."""

    file_hash = sha256_file(pdf_path)

    document_id = (
        f"{domain}_{pdf_path.stem}"
    )

    page_records: list[PageRecord] = []

    with pymupdf.open(pdf_path) as document:

        metadata = document.metadata or {}

        title = metadata.get("title") or None

        for page_number, page in enumerate(
            document,
            start=1,
        ):

            result = analyze_page(page)

            (
                text,
                char_count,
                word_count,
                image_count,
                drawing_count,
                full_page_image,
                visual_candidate,
                ocr_candidate,
            ) = result

            rendered_page: str | None = None

            if visual_candidate:
                render_path = (
                    visual_root
                    / domain
                    / document_id
                    / f"page_{page_number:04d}.png"
                )

                render_page(
                    page,
                    render_path,
                )

                rendered_page = str(
                    render_path
                )

            page_records.append(
                PageRecord(
                    domain=domain,
                    document_id=document_id,
                    source_file=pdf_path.name,
                    source_sha256=file_hash,
                    page=page_number,
                    text=text,
                    char_count=char_count,
                    word_count=word_count,
                    image_count=image_count,
                    drawing_count=drawing_count,
                    full_page_image=full_page_image,
                    visual_candidate=visual_candidate,
                    ocr_candidate=ocr_candidate,
                    rendered_page=rendered_page,
                )
            )

    pages_with_text = sum(
        1
        for page in page_records
        if page.text
    )

    visual_pages = sum(
        1
        for page in page_records
        if page.visual_candidate
    )

    ocr_candidates = sum(
        1
        for page in page_records
        if page.ocr_candidate
    )

    total_characters = sum(
        page.char_count
        for page in page_records
    )

    document_record = DocumentRecord(
        domain=domain,
        document_id=document_id,
        source_file=pdf_path.name,
        source_sha256=file_hash,
        title=title,
        page_count=len(page_records),
        pages_with_text=pages_with_text,
        visual_pages=visual_pages,
        ocr_candidates=ocr_candidates,
        total_characters=total_characters,
    )

    return page_records, document_record


def find_pdfs(
    directory: Path,
) -> list[Path]:
    """Recursively find PDFs."""
    return sorted(
        path
        for path in directory.rglob("*.pdf")
        if path.is_file()
    )


def write_jsonl(
    path: Path,
    records: list[PageRecord],
) -> None:
    """Write one JSON object per line."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Multimodal PDF corpus ingestion "
            "for ML/DL RAG."
        )
    )

    parser.add_argument(
        "--ml-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dl-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--visual-dir",
        type=Path,
        default=Path("data/visuals"),
    )

    args = parser.parse_args()

    for directory, name in [
        (args.ml_dir, "Machine Learning"),
        (args.dl_dir, "Deep Learning"),
    ]:
        if not directory.exists():
            raise FileNotFoundError(
                f"{name} directory does not exist: "
                f"{directory}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.visual_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_pages: list[PageRecord] = []
    all_documents: list[DocumentRecord] = []

    corpora = [
        (
            "machine_learning",
            args.ml_dir,
        ),
        (
            "deep_learning",
            args.dl_dir,
        ),
    ]

    for domain, directory in corpora:

        pdf_files = find_pdfs(directory)

        print()
        print(
            f"{domain}: {len(pdf_files)} PDFs"
        )

        for index, pdf_path in enumerate(
            pdf_files,
            start=1,
        ):

            print(
                f"[{index}/{len(pdf_files)}] "
                f"{pdf_path.name}"
            )

            pages, document = extract_document(
                pdf_path=pdf_path,
                domain=domain,
                visual_root=args.visual_dir,
            )

            all_pages.extend(pages)
            all_documents.append(document)

            print(
                f"    pages={document.page_count} "
                f"visual={document.visual_pages} "
                f"OCR-candidates={document.ocr_candidates}"
            )

    pages_output = (
        args.output_dir
        / "corpus_pages.jsonl"
    )

    manifest_output = (
        args.output_dir
        / "corpus_manifest.json"
    )

    write_jsonl(
        pages_output,
        all_pages,
    )

    manifest = {
        "corpus": {
            "documents": len(all_documents),
            "pages": len(all_pages),
            "characters": sum(
                page.char_count
                for page in all_pages
            ),
            "visual_pages": sum(
                page.visual_candidate
                for page in all_pages
            ),
            "ocr_candidates": sum(
                page.ocr_candidate
                for page in all_pages
            ),
        },
        "domains": {
            "machine_learning": 23,
            "deep_learning": 23,
        },
        "documents": [
            asdict(document)
            for document in all_documents
        ],
    }

    with manifest_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 50)
    print("MULTIMODAL CORPUS INGESTION COMPLETE")
    print("=" * 50)
    print(
        f"Documents:       {len(all_documents)}"
    )
    print(
        f"Pages:           {len(all_pages)}"
    )
    print(
        f"Characters:      "
        f"{sum(page.char_count for page in all_pages):,}"
    )
    print(
        f"Visual pages:    "
        f"{sum(page.visual_candidate for page in all_pages)}"
    )
    print(
        f"OCR candidates:  "
        f"{sum(page.ocr_candidate for page in all_pages)}"
    )
    print()
    print(
        f"Page records:    {pages_output}"
    )
    print(
        f"Manifest:        {manifest_output}"
    )


if __name__ == "__main__":
    main()