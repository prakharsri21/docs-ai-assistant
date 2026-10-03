from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question asked by the user",
    )


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    page: int


class CitationValidation(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    validated_citations: list[Citation] = Field(
        default_factory=list
    )


class RAGPipelineResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(
        default_factory=list
    )
    citation_validation: CitationValidation


class ChatResponse(BaseModel):
    query: str
    basic_rag: RAGPipelineResponse
    hybrid_rag: RAGPipelineResponse
    okf_hybrid_rag: RAGPipelineResponse
