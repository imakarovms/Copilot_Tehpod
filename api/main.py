# FastAPI, эндпоинт POST /diagnose
"""
api/main.py — FastAPI эндпоинт для диагностики инцидентов.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.retriever import Retriever
from src.reranker import Reranker
from src.generator import Generator

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
    retrieved_tickets: list[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация компонентов при старте приложения."""
    logger.info("Инициализация RAG-пайплайна...")
    app.state.retriever = Retriever()
    app.state.reranker = Reranker()
    app.state.generator = Generator()
    logger.info("RAG-пайплайн готов")
    yield


app = FastAPI(
    title="Incident Copilot API",
    description="API для диагностики технических инцидентов на основе RAG",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    """
    Диагностика инцидента.

    - **query**: описание проблемы
    - **top_k**: количество тикетов для поиска (по умолчанию 3)
    """
    try:
        # 1. Поиск
        candidates = app.state.retriever.search(request.query, k=10)

        # 2. Реранкинг
        top_tickets = app.state.reranker.rerank(
            request.query, candidates, top_k=request.top_k
        )

        # 3. Генерация ответа
        result = app.state.generator.generate(request.query, top_tickets)

        return DiagnoseResponse(
            answer=result["answer"],
            citations=result["citations"],
            confidence=result["confidence"],
            retrieved_tickets=top_tickets,
        )

    except Exception as e:
        logger.error("Ошибка при диагностике: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    """Проверка работоспособности API."""
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
