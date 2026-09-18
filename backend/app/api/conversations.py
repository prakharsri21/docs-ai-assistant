from fastapi import APIRouter


router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["conversations"],
)


@router.get("")
async def get_conversations():
    return [
        {
            "id": "demo-1",
            "title": "Introduction",
        },
        {
            "id": "demo-2",
            "title": "Module 3 questions",
        },
    ]