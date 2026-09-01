"""
eval/run_eval_generation.py — Оценка качества генерации ответов (LLM-as-a-Judge через Ollama).
"""
import json
import logging
import sys
import requests
from pathlib import Path

import os
import warnings

# 1. Отключаем предупреждения PyTorch о "старом драйвере" (это не мешает работе)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 2. Отключаем прогресс-бары и логи Hugging Face
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import logging
# Глушим логи конкретных шумных библиотек
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("security.pipeline_security").setLevel(logging.WARNING) # Чтобы не спамил про отсутствующие ссылки

# Добавляем корень проекта в путь
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import IncidentPipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

JUDGE_PROMPT = """Ты справедливый оценщик качества ответов технической поддержки.
Твоя задача: определить, содержит ли сгенерированный ответ ту же основную причину и решение, что и эталонный диагноз.

Эталонный диагноз: "{ground_truth}"
Сгенерированный ответ: "{generated_answer}"

КРИТЕРИИ ОЦЕНКИ:
- ДА: если ключевая причина и рекомендуемое действие сохранены (допускаются перефразирования, другой стиль изложения, дополнительные детали из тикетов).
- НЕТ: если причина или решение фундаментально отличаются, или если ответ "INSUFFICIENT DATA" при наличии конкретного решения в эталоне.

Ответь ТОЛЬКО одним словом: ДА или НЕТ."""


def evaluate_generation():
    eval_file = Path("data/eval/eval.json")
    if not eval_file.exists():
        print(f"Ошибка: файл {eval_file} не найден.")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Фильтруем только те записи, где есть ground_truth
    eval_data = [item for item in eval_data if item.get("ground_truth")]

    if not eval_data:
        print("В eval.json нет записей с полем 'ground_truth'. Добавьте их для оценки.")
        return

    print(f"Загрузка пайплайна для оценки генерации ({len(eval_data)} запросов)...")
    print("Это может занять несколько минут (первый запрос загружает модель в GPU)...\n")

    pipeline = IncidentPipeline()
    print("Пайплайн готов. Начинаем оценку...\n")

    results = []
    yes_count = 0

    for i, item in enumerate(eval_data, 1):
        query = item["description"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{len(eval_data)}] Запрос: {query[:60]}...")

        # 1. Получаем ответ от нашего пайплайна
        result = pipeline.run(query, top_k=3)
        generated_answer = result.get("answer", "")

        # Если пайплайн честно сказал "не знаю", а ground_truth существует — это минус
        if "INSUFFICIENT" in generated_answer.upper():
            judge_verdict = "НЕТ"
        else:
            # 2. Запускаем LLM-as-a-Judge через Ollama API
            judge_prompt_formatted = JUDGE_PROMPT.format(
                ground_truth=ground_truth,
                generated_answer=generated_answer
            )

            messages = [
                {"role": "system", "content": "Ты строгий оценщик. Отвечай только ДА или НЕТ."},
                {"role": "user", "content": judge_prompt_formatted}
            ]

            try:
                resp = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL,
                        "messages": messages,
                        "options": {"temperature": 0.0, "num_predict": 10},
                        "stream": False
                    },
                    timeout=30
                )
                resp.raise_for_status()
                verdict_text = resp.json()["message"]["content"].strip().upper()
                judge_verdict = "ДА" if "ДА" in verdict_text else "НЕТ"
            except Exception as e:
                print(f"  -> Ошибка вызова судьи: {e}")
                judge_verdict = "НЕТ"

        if judge_verdict == "ДА":
            yes_count += 1

        results.append({
            "query": query,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer[:150] + "..." if len(generated_answer) > 150 else generated_answer,
            "verdict": judge_verdict
        })

        print(f"  -> Вердикт: {judge_verdict}\n")

    # 3. Итоговая статистика
    accuracy = (yes_count / len(eval_data)) * 100

    print("=" * 60)
    print("📊 РЕЗУЛЬТАТЫ GENERATION EVAL")
    print("=" * 60)
    print(f"Всего оценено запросов: {len(eval_data)}")
    print(f"Совпадений с эталоном (ДА): {yes_count}")
    print(f"Точность генерации (Generation Accuracy): {accuracy:.2f}%")
    print("=" * 60)

    # Сохраняем детальный отчет
    report_path = Path("eval/results/generation_eval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Детальный отчет сохранен в: {report_path}")


if __name__ == "__main__":
    evaluate_generation()