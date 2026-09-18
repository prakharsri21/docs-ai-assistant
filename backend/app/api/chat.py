from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import generate_chat_response


router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return generate_chat_response(request.message)