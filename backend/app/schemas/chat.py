from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question asked by the user",
    )


class Source(BaseModel):
    module: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []