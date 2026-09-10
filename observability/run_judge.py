"""
observability/run_judge.py — CLI-скрипт для запуска независимого судьи.

Использование:
    python observability/run_judge.py --limit 10
    python observability/run_judge.py --auto
"""

import sys
import os
import argparse
import logging

# Явно добавляем корень проекта в путь поиска модулей.
# Это решает проблему "No module named 'observability'" при прямом запуске скрипта.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Теперь абсолютные импорты будут работать корректно
from observability.db import get_connection
from observability.judge import evaluate_call_from_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_unevaluated_calls(limit: int = 10) -> list:
    """Получает ID неоценённых вызовов из БД."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id FROM calls 
        WHERE is_evaluated = 0 
          AND success = 1
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    
    call_ids = [row["id"] for row in cursor.fetchall()]
    conn.close()
    
    return call_ids


def run_judge(limit: int = 10):
    """Оценивает последние N неоценённых вызовов."""
    call_ids = get_unevaluated_calls(limit)
    
    if not call_ids:
        logger.info("Нет неоценённых вызовов")
        return
    
    logger.info(f"Найдено {len(call_ids)} неоценённых вызовов")
    
    success_count = 0
    error_count = 0
    
    for call_id in call_ids:
        try:
            result = evaluate_call_from_db(call_id)
            if result["score"] is not None:
                success_count += 1
                logger.info(f"  {call_id[:8]}... -> {result['score']}/5")
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            logger.error(f"  {call_id[:8]}... -> Ошибка: {e}")
    
    logger.info(f"Готово: {success_count} оценено, {error_count} ошибок")


def main():
    parser = argparse.ArgumentParser(description="Независимый судья для оценки качества LLM")
    parser.add_argument("--limit", type=int, default=10, help="Количество вызовов для оценки")
    parser.add_argument("--auto", action="store_true", help="Оценить все неоценённые вызовы")
    
    args = parser.parse_args()
    
    limit = 1000 if args.auto else args.limit
    run_judge(limit=limit)


if __name__ == "__main__":
    main()