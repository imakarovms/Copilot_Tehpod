.PHONY: help install setup index api interactive diagnose test test-security eval-retrieval eval-generation eval-all clean clean-all ollama-serve ollama-pull

# ============================================================================
# Incident Copilot - Makefile
# Стек: Python 3.12, FastAPI, ChromaDB, BM25, Ollama (Qwen2.5-7B), DiskCache
# ============================================================================

PYTHON = python3
VENV = venv
ACTIVATE = . $(VENV)/bin/activate
APP_MODULE = api.main:app

# ============================================================================
# Помощь
# ============================================================================

help: ## Показать список доступных команд
	@echo ""
	@echo "    install          - Создать venv и установить зависимости"
	@echo "    ollama-pull      - Скачать модель Qwen 2.5 7B в Ollama"
	@echo "    index            - Построить индекс тикетов в ChromaDB"
	@echo "    setup            - Полная настройка (install + ollama-pull + index)"
	@echo ""
	@echo "    ollama-serve     - Запустить Ollama в фоновом режиме (тихий режим)"
	@echo "    interactive      - Интерактивный CLI (модели грузятся один раз)"
	@echo "    api              - Запустить FastAPI сервер (Swagger: http://localhost:8000/docs)"
	@echo "    diagnose         - Одноразовая диагностика: make diagnose ARGS='текст проблемы'"
	@echo ""
	@echo "    test             - Запустить все unit-тесты"
	@echo "    test-security    - Запустить только тесты безопасности"
	@echo "    eval-retrieval   - Оценить качество поиска (Hit Rate @ 3)"
	@echo "    eval-generation  - Оценить качество генерации (LLM-as-a-Judge)"
	@echo "    eval-all         - Запустить все оценки подряд"
	@echo ""
	@echo "    clean            - Удалить временные файлы и кеш"
	@echo "    clean-all        - Удалить всё, включая venv, БД и модели (ВНИМАНИЕ!)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Установка и настройка
# ============================================================================

install: ## Создать виртуальное окружение и установить зависимости
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt
	@echo "✓ Окружение установлено"

ollama-pull: ## Скачать модель Qwen 2.5 7B в Ollama
	@echo "Загрузка модели в Ollama (может занять 1-2 минуты)..."
	ollama pull qwen2.5:7b
	@echo "✓ Модель загружена"

index: ## Построить индекс тикетов в ChromaDB
	$(ACTIVATE) && $(PYTHON) scripts/build_index.py
	@echo "✓ Индекс ChromaDB построен"

setup: install ollama-pull index ## Полная настройка проекта с нуля

# ============================================================================
# Управление Ollama
# ============================================================================

ollama-serve: ## Запустить Ollama в фоновом режиме (без спама в консоль)
	@echo "Запуск Ollama..."
	@pkill ollama || true
	@OLLAMA_LOG_LEVEL=error ollama serve > /dev/null 2>&1 &
	@sleep 2
	@echo "✓ Ollama запущен в фоне. Проверьте: ollama ps"

# ============================================================================
# Запуск приложения
# ============================================================================

interactive: ## Запустить интерактивный CLI (рекомендуется для демо)
	$(ACTIVATE) && $(PYTHON) -m cli.interactive

api: ## Запустить FastAPI сервер с автоперезагрузкой
	$(ACTIVATE) && uvicorn $(APP_MODULE) --reload --host 0.0.0.0 --port 8000

diagnose: ## Одноразовая диагностика: make diagnose ARGS="не работает VPN"
	$(ACTIVATE) && $(PYTHON) -m cli.diagnose "$(ARGS)"

# ============================================================================
# Тестирование и Оценка (согласно ТЗ)
# ============================================================================

test: ## Запустить все тесты
	$(ACTIVATE) && pytest tests/ -v

test-security: ## Запустить только тесты безопасности (PII + инъекции)
	$(ACTIVATE) && pytest tests/test_security.py -v

eval-retrieval: ## Оценить качество поиска (Retrieval Hit Rate)
	$(ACTIVATE) && $(PYTHON) eval/run_eval.py
	$(ACTIVATE) && $(PYTHON) eval/run_eval_rerank.py

eval-generation: ## Оценить качество генерации (LLM-as-a-Judge)
	$(ACTIVATE) && $(PYTHON) eval/run_eval_generation.py

eval-all: eval-retrieval eval-generation ## Запустить полную оценку системы
	@echo "========================================"
	@echo "✓ Все этапы оценки завершены"
	@echo "========================================"

# ============================================================================
# Очистка
# ============================================================================

clean: ## Удалить временные файлы, кеш и результаты eval
	rm -rf __pycache__/ .pytest_cache/ eval/results/ cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Временные файлы и кеш удалены"

clean-all: clean ## Удалить ВСЁ: venv, ChromaDB, модели (НЕОБРАТИМО)
	rm -rf $(VENV)/ chroma_db/ models/
	@echo "✓ Полная очистка выполнена. Требуется повторный 'make setup'"