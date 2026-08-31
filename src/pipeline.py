"""
src/pipeline.py — End-to-End пайплайн.
"""
import logging
from src.retriever import Retriever
from src.reranker import Reranker
from src.generator import Generator
from security import SecurityValidator
from cache.cache import ResponseCache

logger = logging.getLogger(__name__)


class IncidentPipeline:
    def __init__(self):
        logger.info("Инициализация IncidentPipeline...")
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = Generator()
        self.security = SecurityValidator()
        self.cache = ResponseCache()

        logger.info("IncidentPipeline готов.")

    def run(self, query: str, top_k: int = 3) -> dict:
        #  Проверяем кеш
        cached = self.cache.get(query, top_k)
        if cached is not None:
            cached["from_cache"] = True
            return cached
        
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
                "from_cache": False,
            }

        # 2. Поиск
        candidates = self.retriever.search(sec["redacted_text"], k=10)

        # 3. Реранкинг
        top_tickets = self.reranker.rerank(
            sec["redacted_text"], candidates, top_k=top_k
        )

        # 4. Генерация
        result = self.generator.generate(sec["redacted_text"], top_tickets)

        # Определяем 4 уровня уверенности на основе качества поиска и ответа
        if not top_tickets or top_tickets[0].get("score", 0) < 0.3:
            confidence_level = "недостаточно похожих случаев"
        elif result["answer"].upper().startswith("INSUFFICIENT"):
            confidence_level = "низкая"
        elif len(result["citations"]) >= 2 and result["citations"][0] in result["answer"]:
            confidence_level = "высокая"
        else:
            confidence_level = "средняя"

        final_result = {
            "query": query,
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence": confidence_level,
            "risk_score": result.get("risk_score", 0.0),
            "retrieved_tickets": top_tickets,
            "blocked": False,
            "from_cache": False,
        }

        # 7. Сохраняем в кеш (только успешные ответы)
        if not result["answer"].upper().startswith("INSUFFICIENT"):
            self.cache.set(query, top_k, final_result)

        return final_result