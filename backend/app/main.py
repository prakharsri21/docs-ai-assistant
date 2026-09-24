from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router


app = FastAPI(
    title="Course AI Assistant API",
    version="0.1.0",
)

# Allow our local Next.js frontend to communicate with FastAPI.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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