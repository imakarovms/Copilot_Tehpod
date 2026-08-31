"""
cli/diagnose.py — CLI интерфейс для диагностики инцидентов.
"""
import sys
import logging
import traceback
from pathlib import Path

# Принудительно включаем логирование в консоль
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("DEBUG: Скрипт cli.diagnose запущен")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("ОШИБКА: Не указан запрос.")
        print("Использование: python -m cli.diagnose 'текст вашей проблемы'")
        sys.exit(1)
        
    query = sys.argv[1]
    print(f"DEBUG: Получен запрос: '{query}'")
    
    try:
        print("DEBUG: Импортируем IncidentPipeline...")
        from src.pipeline import IncidentPipeline
        
        print("DEBUG: Инициализируем пайплайн (это может занять 10-20 сек)...")
        pipeline = IncidentPipeline()
        
        print(f"DEBUG: Запускаем обработку запроса...")
        result = pipeline.run(query, top_k=3)
        
        print("\n" + "=" * 60)
        print("🔍 ЗАПРОС:", result.get("query"))
        print("-" * 60)
        
        if result.get("blocked"):
            print("⛔ ЗАПРОС ЗАБЛОКИРОВАН СИСТЕМОЙ БЕЗОПАСНОСТИ")
            print(f"Причина: {result.get('answer')}")
            print(f"Уровень риска: {result.get('risk_score')}")
        else:
            print("💡 ДИАГНОЗ И РЕШЕНИЕ:")
            print(result.get("answer"))
            print("\n📎 Источники:", ", ".join(result.get("citations", [])))
            print(f"📊 Уверенность: {result.get('confidence', 'unknown').upper()}")
            
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "!" * 60)
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        print("Полный стек вызовов:")
        traceback.print_exc()
        print("!" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()