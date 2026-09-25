from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models import Conversation
from pydantic import BaseModel, ConfigDict


router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["conversations"],
)


class ConversationCreate(BaseModel):
    user_id: int
    title: str | None = None


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    title: str | None
    model_config = ConfigDict(from_attributes=True)


@router.post("", response_model=ConversationResponse)
def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db),
):
    conversation = Conversation(
        user_id=request.user_id,
        title=request.title,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation


@router.get("", response_model=list[ConversationResponse])
def get_conversations(
    user_id: int,
    db: Session = Depends(get_db),
):
    statement = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
    )

    conversations = db.scalars(statement).all()

    return conversations