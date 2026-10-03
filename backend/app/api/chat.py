from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import (
    generate_chat_response,
    stream_chat_response,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    return generate_chat_response(
        request.message
    )


@router.post(
    "/chat/stream",
)
def chat_stream(
    request: ChatRequest,
):
    return StreamingResponse(
        stream_chat_response(
            request.message
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )