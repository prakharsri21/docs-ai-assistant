from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router


app = FastAPI(
    title="Course AI Assistant API",
    version="0.1.0",
)


app.include_router(chat_router)
app.include_router(conversations_router)


@app.get("/")
async def root():
    return {
        "message": "Course AI Assistant API is running"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }