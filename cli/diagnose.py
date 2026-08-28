# простой CLI: python -m cli.diagnose "текст проблемы"# cli/diagnose.py
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retriever import Retriever
from src.reranker import Reranker
from src.generator import Generator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def main():
    if len(sys.argv) < 2:
        print("Использование: python -m cli.diagnose 'текст вашей проблемы'")
        sys.exit(1)

    query = sys.argv[1]
    print(f"\n🔍 Запрос: {query}\n" + "-" * 50)

    # 1. Поиск
    retriever = Retriever()
    candidates = retriever.search(query, k=5)

    # 2. Реранкинг
    reranker = Reranker()
    top_3 = reranker.rerank(query, candidates, top_k=3)

    print(f"Найдено релевантных тикетов: {', '.join([t['id'] for t in top_3])}")

    # 3. Генерация
    generator = Generator()
    result = generator.generate(query, top_3)

    print("\n💡 ДИАГНОЗ И РЕШЕНИЕ:")
    print(result["answer"])
    print(f"\n📎 Источники: {', '.join(result['citations'])}")
    print(f"📊 Уверенность: {result['confidence'].upper()}")


if __name__ == "__main__":
    main()
