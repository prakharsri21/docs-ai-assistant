from app.schemas.chat import ChatResponse


def generate_chat_response(message: str) -> ChatResponse:
    """
    Temporary chat logic.

    Later this function will:
    1. Check safety/scope
    2. Search the course knowledge base
    3. Send retrieved context to the LLM
    4. Generate citations
    """

    return ChatResponse(
        answer=f"You asked: {message}",
        sources=[],
    )