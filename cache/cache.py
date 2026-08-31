"""
src/cache.py — кеш ответов пайплайна на основе DiskCache.
"""
import hashlib
import json
import logging
from pathlib import Path

import diskcache

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache/responses")
CACHE_TTL = 3600  # 1 час
MAX_CACHE_SIZE = 1000  # Максимум записей


class ResponseCache:
    """Кеш ответов пайплайна. Кеширует полный результат по хешу запроса."""

    def __init__(self, ttl: int = CACHE_TTL, max_size: int = MAX_CACHE_SIZE):
        self.cache = diskcache.Cache(str(CACHE_DIR))
        self.ttl = ttl
        self.max_size = max_size
        self._evict_if_needed()
        logger.info("Кеш инициализирован: %s (TTL=%ds, max=%d)", CACHE_DIR, ttl, max_size)

    def _make_key(self, query: str, top_k: int) -> str:
        """Создает уникальный ключ кеша из запроса."""
        normalized = query.strip().lower()
        raw = f"{normalized}|top_k={top_k}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, top_k: int) -> dict | None:
        """Получает результат из кеша. Возвращает None при промахе."""
        key = self._make_key(query, top_k)
        result = self.cache.get(key)
        if result is not None:
            logger.info("Cache HIT для запроса: '%s...'", query[:40])
            return result
        logger.debug("Cache MISS для запроса: '%s...'", query[:40])
        return None

    def set(self, query: str, top_k: int, result: dict) -> None:
        """Сохраняет результат в кеш."""
        key = self._make_key(query, top_k)
        self.cache.set(key, result, expire=self.ttl)
        logger.debug("Сохранено в кеш: ключ=%s", key[:16])

    def _evict_if_needed(self):
        """Удаляет старые записи, если кеш переполнен."""
        if len(self.cache) > self.max_size:
            # DiskCache автоматически удаляет oldest при переполнении
            # но мы можем явно очистить expired
            self.cache.expire()
            logger.info("Очистка expired записей из кеша")

    def clear(self):
        """Полностью очищает кеш."""
        self.cache.clear()
        logger.info("Кеш полностью очищен")

    def stats(self) -> dict:
        """Возвращает статистику кеша."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "directory": str(CACHE_DIR),
        }