"""
src/pipeline.py — End-to-End пайплайн.
"""
import logging
from src.retriever import Retriever
from src.reranker import Reranker
from src.generator import Generator
from security import SecurityValidator

logger = logging.getLogger(__name__)


class IncidentPipeline:
    def __init__(self):
        logger.info("Инициализация IncidentPipeline...")
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = Generator()
        self.security = SecurityValidator()
        logger.info("IncidentPipeline готов.")

    def run(self, query: str, top_k: int = 3) -> dict:
        # 1. Проверка запроса
        sec = self.security.validate_query(query)
        if not sec["safe"]:
            return {
                "query": query,
                "answer": f"Запрос отклонён: {sec['label']}",
                "citations": [],
                "confidence": "low",
                "risk_score": sec["risk_score"],
                "blocked": True,
            }

        # 2. Поиск
        candidates = self.retriever.search(sec["redacted_text"], k=10)

        # 3. Реранкинг
        top_tickets = self.reranker.rerank(
            sec["redacted_text"], candidates, top_k=top_k
        )

        # 4. Генерация
        result = self.generator.generate(sec["redacted_text"], top_tickets)

        return {
            "query": query,
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "risk_score": result.get("risk_score", 0.0),
            "retrieved_tickets": top_tickets,
            "blocked": False,
        }