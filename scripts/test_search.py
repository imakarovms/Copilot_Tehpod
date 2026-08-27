# scripts/test_search.py
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для корректных импортов
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    retriever = Retriever()
    
    # Тестовые запросы (подставьте свои, если датасет специфичный)
    queries = [
        "Не работает интернет, VPN постоянно отваливается",
        "Принтер жует бумагу и ничего не печатает",
        "Не могу зайти в почту, пишет что пароль неверный"
    ]
    
    for q in queries:
        print(f"\n🔍 Запрос: '{q}'")
        results = retriever.search(q, k=3)
        
        if not results:
            print("  Ничего не найдено.")
            continue
            
        for i, res in enumerate(results, 1):
            title = res.get('title', 'Без заголовка')
            category = res.get('category', 'N/A')
            score = res['score']
            print(f"  {i}. [Score: {score:.3f}] {title} (ID: {res['id']})")
            print(f"     Категория: {category}")

if __name__ == "__main__":
    main()