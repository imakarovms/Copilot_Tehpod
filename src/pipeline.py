# сборка всего пайплайна end-to-end
"""
src/pipeline.py — End-to-End пайплайн диагностики инцидентов.
"""
import logging
from src.retriever import Retriever
from src.reranker import Reranker
from src.generator import Generator
from src.security import SecurityValidator

logger = logging.getLogger(__name__)


class IncidentPipeline:
    """
    Единая точка входа для обработки запроса пользователя.
    Инициализирует все компоненты и управляет потоком данных.
    """

    def __init__(self):
        logger.info("Инициализация IncidentPipeline...")
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = Generator()
        logger.info("IncidentPipeline готов к работе.")

    def run(self, query: str, top_k: int = 3) -> dict:
        """
        Полный цикл обработки запроса.

        Args:
            query: Текст проблемы от пользователя.
            top_k: Количество тикетов для возврата в ответе.

        Returns:
            Словарь с ответом, цитатами, уверенностью и метаданными.
        """
        # 1. Валидация ввода
        safe_query = SecurityValidator.validate_query(query)

        # 2. Гибридный поиск (берем с запасом для реранкера)
        candidates = self.retriever.search(safe_query, k=10)

        # 3. Реранкинг кандидатов
        top_tickets = self.reranker.rerank(safe_query, candidates, top_k=top_k)

        # 4. Генерация ответа через LLM (с внутренней валидацией вывода)
        result = self.generator.generate(safe_query, top_tickets)

        return {
            "query": safe_query,
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "retrieved_tickets": top_tickets,
        }
