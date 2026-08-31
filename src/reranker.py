"""
src/reranker.py — легковесный эвристический реранкер.
Используется вместо тяжелого Cross-Encoder для экономии ресурсов и времени.
"""
import logging

logger = logging.getLogger(__name__)

CATEGORY_KEYWORDS = {
    "network": ["vpn", "интернет", "сеть", "сайт", "dns", "видеозвонок", "фриз"],
    "access": ["пароль", "вход", "доступ", "аккаунт", "sso", "авториз"],
    "performance": ["дашборд", "график", "зависа", "медленн", "экспорт", "100%"],
    "billing": ["списан", "оплат", "деньг", "подписк", "тариф"],
    "integrations": ["slack", "telegram", "api", "sync", "синхрониз"],
}

def detect_category(query: str) -> str | None:
    query_lower = query.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return category
    return None

class Reranker:
    def __init__(self):
        logger.info("Инициализирован эвристический реранкер (без ML-моделей)")

    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
        if not documents:
            return []

        query_category = detect_category(query)
        reranked = []

        for doc in documents:
            new_doc = doc.copy()
            base_score = new_doc.get("score", 0.0)
            
            # Эвристический бонус: +0.02 за совпадение категории
            if query_category and doc.get("category") == query_category:
                new_doc["rerank_score"] = base_score + 0.02
            else:
                new_doc["rerank_score"] = base_score
                
            reranked.append(new_doc)

        # Сортируем по новому скору
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]