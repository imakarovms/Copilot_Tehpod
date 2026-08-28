"""
src/reranker.py — точный реранкинг кандидатов с помощью Cross-Encoder.
"""
import logging
from sentence_transformers import CrossEncoder

from config.settings import settings

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self):
        logger.info("Загрузка модели реранкинга: %s", settings.reranker_model)
        self.model = CrossEncoder(settings.reranker_model)
        logger.info("Модель реранкинга загружена")

    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
        """
        Переранжирует список документов относительно запроса.
        
        Args:
            query: текст запроса.
            documents: список кандидатов от Retriever (должен содержать ключ "document").
            top_k: сколько лучших вернуть.
            
        Returns:
            Отсортированный список документов с новым полем "rerank_score".
        """
        if not documents:
            return []

        # Формируем пары (запрос, текст_документа) для кросс-энкодера
        pairs = [(query, doc.get("document", "")) for doc in documents]
        
        # Модель возвращает логиты (чем выше, тем релевантнее)
        scores = self.model.predict(pairs)
        
        # Прикрепляем скоры и сортируем по убыванию
        reranked = []
        for doc, score in zip(documents, scores):
            new_doc = doc.copy()
            new_doc["rerank_score"] = float(score)
            reranked.append(new_doc)
            
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]