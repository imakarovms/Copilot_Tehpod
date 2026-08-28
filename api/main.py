"""
api/main.py — FastAPI эндпоинт для диагностики инцидентов.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import IncidentPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DiagnoseRequest(BaseModel):
    query: str
    top_k: int = 3


class DiagnoseResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация пайплайна при старте приложения."""
    logger.info("Загрузка IncidentPipeline...")
    app.state.pipeline = IncidentPipeline()
    logger.info("Pipeline готов")
    yield


app = FastAPI(
    title="Incident Copilot API",
    description="API для диагностики технических инцидентов на основе RAG",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    try:
        result = app.state.pipeline.run(request.query, top_k=request.top_k)
        return DiagnoseResponse(
            answer=result["answer"],
            citations=result["citations"],
            confidence=result["confidence"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Ошибка пайплайна: %s", str(e))
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера") from e


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}
