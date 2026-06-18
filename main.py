from fastapi import FastAPI
from app.core.logging import logger

app = FastAPI(title="AI Scientific Intelligence Platform")


@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "ok"}