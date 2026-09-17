from fastapi import FastAPI

app = FastAPI(
    title="Legal ai assistant API",
    version="0.1.0"
)

@app.get("/")
async def root():    
    return {
        "message": "legal ai assistant API is running"
        }

@app.get("/health")
async def health():
    return {
        "status": "healthy"
        }