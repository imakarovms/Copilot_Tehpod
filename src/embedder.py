"""
src/embedder.py — эмбеддинги тикетов и запросов.

КАК текст превращается в вектор:
- какая модель (settings.embedding_model)
- по какому шаблону тикет превращается в текст (ticket_to_text)
- нормализация (normalize_embeddings=True → поиск = dot product)
"""
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Возвращает модель, при первом вызове — загружает с кэша HuggingFace."""
    global _model
    if _model is None:
        logger.info("Загружаю модель эмбеддингов: %s ...", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        _model.max_seq_length = settings.embedding_max_seq_length
        logger.info(
            "Модель загружена. Размерность: %d, max_seq_length: %d",
            _model.get_embedding_dimension(),
            _model.max_seq_length,
        )
    return _model


def ticket_to_text(ticket: dict) -> str:
    """
    Тикет → текст для эмбеддинга.

    (title + description)
    """
    title = (ticket.get("title") or "").strip()
    description = (ticket.get("description") or "").strip()

    # У eval-тикетов нет title — работаем и с ними (нужен для run_eval).
    if title and description:
        return f"{title}. {description}"
    return description or title


def embed_texts(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """
    Эмбеддинг списка текстов.

    Возвращает np.ndarray формы (len(texts), embedding_dim), L2-нормализованный:
    после нормализации косинусная близость = скалярное произведение,
    поиск по всей базе — одно матричное умножение.

    Args:
        texts: список текстов (непустых строк).
        batch_size: размер батча; None → из конфига.

    Raises:
        ValueError: если texts пуст или содержит не-строки.
    """
    if not texts:
        raise ValueError("embed_texts: передан пустой список текстов")
    if not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError("embed_texts: все элементы должны быть непустыми строками")

    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size or settings.embedding_batch_size,
        normalize_embeddings=True,   # ключевое: косинус == dot product
        convert_to_numpy=True,       # явно возвращаем numpy, не torch-тензоры
        show_progress_bar=len(texts) > 50,  # прогрессбар только на больших батчах
    )

    result = np.asarray(embeddings, dtype=np.float32)

    # Самопроверка контракта: форма и нормализация. Дёшево и ловит
    # ситуацию "случайно передали тексты из другой модели".
    expected_dim = settings.embedding_dim
    if result.shape != (len(texts), expected_dim):
        raise RuntimeError(
            f"Модель вернула вектор размерности {result.shape[1]}, "
            f"ожидалось {expected_dim}. Проверьте embedding_dim в конфиге — "
            f"он должен соответствовать settings.embedding_model."
        )
    norms = np.linalg.norm(result, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError(
            "Векторы не нормализованы (||v|| != 1). "
            "Не отключайте normalize_embeddings — иначе dot product "
            "перестаёт быть косинусной близостью."
        )

    logger.debug("Зачембдено %d текстов, форма: %s", len(texts), result.shape)
    return result


def embed_query(query: str) -> np.ndarray:
    """
    Эмбеддинг одного запроса (описания новой проблемы).

    Возвращает вектор формы (embedding_dim,), нормализованный.
    Тот же путь кодирования, что и у тикетов — иначе близость мусорная.
    """
    if not query or not query.strip():
        raise ValueError("embed_query: запрос пуст")
    return embed_texts([query.strip()])[0]


# ---------------------------------------------------------------------------
# Утилиты близости. Обёртки с понятными именами: остальной код проекта
# пишет cosine_similarity(...), а не вспоминает, что там dot product.
# ---------------------------------------------------------------------------
def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Близость одного нормализованного вектора к матрице нормализованных.

    Args:
        query:  (dim,) — вектор запроса.
        matrix: (n, dim) — векторы тикетов.

    Returns:
        (n,) — косинусные близости, каждая в диапазоне ~[-1, 1].
    """
    return matrix @ query


def top_k_by_similarity(
    query: np.ndarray, matrix: np.ndarray, k: int
) -> list[tuple[int, float]]:
    """
    Топ-k индексов и скоров по убыванию близости.

    Returns:
        [(index, score), ...] — длина min(k, n).
    """
    scores = cosine_similarity(query, matrix)
    k = min(k, len(scores))
    # argpartition — O(n), быстрее полной сортировки; сортируем только топ-k
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Загрузка тикетов из файла. Живёт здесь, а не в indexer, потому что
# indexer и eval/run_eval оба нуждаются в одном способе чтения.
# ---------------------------------------------------------------------------
def load_tickets(path: str | Path) -> list[dict]:
    """Читает JSON-массив тикетов, валидирует обязательные поля."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл тикетов не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        tickets = json.load(f)

    if not isinstance(tickets, list) or not tickets:
        raise ValueError(f"Файл {path} должен содержать непустой JSON-массив")

    required = {"id", "category", "description"}
    for i, t in enumerate(tickets):
        missing = required - set(t.keys())
        if missing:
            raise ValueError(
                f"Тикет #{i} в {path} не содержит поля: {missing}. "
                f"Тикет: {str(t)[:100]}..."
            )

    ids = [t["id"] for t in tickets]
    if len(ids) != len(set(ids)):
        dupes = {x for x in ids if ids.count(x) > 1}
        raise ValueError(f"Дубликаты id в {path}: {dupes}")

    logger.info("Загружено %d тикетов из %s", len(tickets), path)
    return tickets