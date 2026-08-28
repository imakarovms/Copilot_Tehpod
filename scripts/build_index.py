# scripts/build_index.py
import logging
import sys
from pathlib import Path

# Гарантируем, что импорты src/ и config/ работают при любом способе запуска
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings
from src.embedder import load_tickets
from src.indexer import Indexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    # Путь к данным, которые должны пойти в индекс
    index_path = Path(settings.data_dir) / "index" / "tickets.json"

    if not index_path.exists():
        logger.error("Файл %s не найден!", index_path)
        logger.info(
            "Создайте его вручную для теста или запустите scripts/generate_dataset.py"
        )
        sys.exit(1)

    logger.info("Читаем тикеты из %s ...", index_path)
    tickets = load_tickets(index_path)

    logger.info("Инициализируем ChromaDB ...")
    indexer = Indexer()

    # Если хотите гарантированно пересобрать индекс с нуля, раскомментируйте:
    # indexer.clear()

    logger.info("Индексируем %d тикетов ...", len(tickets))
    indexer.add_tickets(tickets)

    # Финальная валидация (критерий готовности из плана)
    db_count = indexer.get_count()
    if db_count != len(tickets):
        logger.error(
            "ПРОВАЛ: в базе %d тикетов, а ожидалось %d", db_count, len(tickets)
        )
        sys.exit(1)

    logger.info("ГОТОВО! В коллекции %d тикетов. Индекс построен.", db_count)


if __name__ == "__main__":
    main()
