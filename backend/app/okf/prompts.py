OKF_EXTRACTION_PROMPT = """
You are building a structured knowledge representation from
machine learning and deep learning course material.

Convert the supplied source chunk into ONE knowledge node.

Rules:

1. Only use information supported by the supplied source.
2. Do not invent facts.
3. Preserve important technical terminology.
4. Identify the main concept or idea represented by the chunk.
5. 5. Create relationships only when the source directly states
   the relationship or describes it through an explicit
   comparison, condition, dependency, or equivalence.

6. Do not infer broad taxonomy such as "is-a", "type-of",
   or "subclass-of" unless the source explicitly supports it.
7. Preserve mathematical notation when present.
8. Keep source provenance exactly as supplied.
9. Do not generate an ID. The application will assign the
   canonical source chunk ID.
10. Prefer a meaningful concept title over a generic title.

The output must conform to the supplied OKF schema.

SOURCE CHUNK:
{chunk_text}

SOURCE METADATA:
Document ID: {document_id}
Source file: {source_file}
Page: {page}
Chunk ID: {chunk_id}
Domain: {domain}
"""