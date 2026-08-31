"""
cli/interactive.py — интерактивный режим работы с Copilot.
Модели загружаются один раз при старте, далее ответы генерируются мгновенно.
"""
import logging
import sys
from pathlib import Path
import time

# Добавляем корень проекта в путь
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import IncidentPipeline

logging.basicConfig(level=logging.WARNING) # Скрываем лишние логи при диалоге

def main():
    print("=" * 60)
    print("Загрузка Incident Copilot... (это займет ~15 секунд один раз)")
    print("=" * 60)
    
    try:
        pipeline = IncidentPipeline()
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Copilot готов к работе!")
    print("Введите описание проблемы или 'выход' / 'quit' для завершения.")
    print("=" * 60)

    while True:
        try:
            query = input("\n👤 Запрос: ").strip()
            
            if not query:
                continue
            if query.lower() in ["выход", "quit", "exit"]:
                print("Завершение работы...")
                break

            print("🤖 Думаю...", end="", flush=True)
            start = time.perf_counter()
            
            # Запускаем пайплайн
            result = pipeline.run(query, top_k=3)
            
            print("\r" + " " * 20 + "\r", end="") # Очистка строки "Думаю..."

            end = time.perf_counter()

            if result.get("blocked"):
                print(f"⛔ Запрос заблокирован: {result.get('answer')}")
                print(f'Время выполнения {end - start:.3f}')
            else:
                print(f"\n💡 Ответ:\n{result.get('answer')}")
                print(f"📎 Источники: {', '.join(result.get('citations', []))}")
                print(f"📊 Уверенность: {result.get('confidence', 'unknown').upper()}")
                print(f'Время выполнения {end - start:.3f}')

        except KeyboardInterrupt:
            print("\n\nЗавершение работы...")
            break
        except Exception as e:
            print(f"\nОшибка при обработке: {e}")

if __name__ == "__main__":
    main()