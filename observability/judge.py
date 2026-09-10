"""
observability/judge.py — независимый судья для оценки качества ответов LLM.

Использует легковесную локальную модель (qwen2.5:1.5b) через Ollama.

"""

import json
import logging
import re
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# Конфигурация судьи
JUDGE_MODEL = "qwen2.5:1.5b"
JUDGE_URL = "http://localhost:11434/api/generate"
JUDGE_TIMEOUT = 30  # секунд

JUDGE_SYSTEM_PROMPT = """
Ты — независимый эксперт по оценке качества ответов технической поддержки.
Твоя задача — оценить ответ AI-ассистента на вопрос пользователя.

Критерии оценки (от 1 до 5):
- 5: Идеальный ответ. Точно отвечает на вопрос, опирается на предоставленный контекст, 
     структурирован (диагноз + шаги решения), нет галлюцинаций.
- 4: Хороший ответ. Отвечает на вопрос, в основном опирается на контекст, 
     но есть мелкие недочёты в структуре или полноте.
- 3: Удовлетворительный ответ. Отвечает на вопрос, но есть проблемы: 
     неполная опора на контекст, слабая структура, или есть незначительные неточности.
- 2: Плохой ответ. Частично отвечает на вопрос, но содержит галлюцинации, 
     игнорирует контекст, или структура хаотична.
- 1: Неприемлемый ответ. Не отвечает на вопрос, содержит серьёзные галлюцинации, 
     или полностью противоречит контексту.

Важно:
- Оценивай СТРОГО на основе предоставленного контекста (исторических тикетов).
- Если ответ содержит "INSUFFICIENT DATA" — это корректный ответ, если контекст действительно не релевантен (оценка 4-5).
- Если ответ выдумывает факты, которых нет в контексте — снижай оценку.

Ответь СТРОГО в формате JSON без дополнительного текста:
{
  "score": <число от 1 до 5>,
  "reasoning": "<краткое обоснование оценки на русском языке, 1-2 предложения>"
}
"""


def _parse_judge_response(response_text: str) -> dict:
    """Извлекает и валидирует JSON из ответа модели."""
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError(f"Не удалось найти JSON в ответе судьи: {response_text}")
    
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Не удалось распарсить JSON: {e}. Текст: {json_str}")
    
    if "score" not in result or "reasoning" not in result:
        raise ValueError(f"Некорректная структура JSON: {result}")
    
    score = float(result["score"])
    if not (1.0 <= score <= 5.0):
        raise ValueError(f"Оценка вне диапазона [1, 5]: {score}")
    
    return {
        "score": score,
        "reasoning": str(result["reasoning"]),
    }


def evaluate_quality(
    input_text: str,
    output_text: str,
    context: Optional[str] = None,
) -> dict:
    """Оценивает качество ответа LLM через локальную модель Ollama."""
    
    user_prompt = f"""
ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{input_text}

ОТВЕТ AI-АССИСТЕНТА:
{output_text}
"""
    
    if context:
        user_prompt += f"""

КОНТЕКСТ (исторические тикеты):
{context}
"""
    
    user_prompt += "\nОцени качество ответа по критериям выше. Ответь СТРОГО в формате JSON."
    
    full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"
    
    try:
        response = requests.post(
            JUDGE_URL,
            json={
                "model": JUDGE_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Детерминированный ответ для консистентности оценок
                    "num_predict": 300   # Ограничиваем длину ответа, так как нужен только JSON
                }
            },
            timeout=JUDGE_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "")
        
        logger.info(f"Судья ({JUDGE_MODEL}) вернул ответ: {response_text[:150]}...")
        return _parse_judge_response(response_text)
        
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Ollama недоступен по адресу {JUDGE_URL}. Запустите 'ollama serve'.")
    except Exception as e:
        logger.error(f"Ошибка при оценке качества: {e}")
        raise


def evaluate_call_from_db(call_id: str) -> dict:
    """Оценивает конкретный вызов из БД и сохраняет результат."""
    from observability.db import get_connection
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT c.id, d.input_text, d.output_text 
        FROM calls c
        LEFT JOIN call_details d ON c.id = d.call_id
        WHERE c.id = ?
    """, (call_id,))
    
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Вызов {call_id} не найден в БД")
    
    input_text = row["input_text"] or ""
    output_text = row["output_text"] or ""
    conn.close()
    
    if not input_text or not output_text:
        logger.warning(f"Вызов {call_id} не содержит текстов для оценки")
        return {"score": None, "reasoning": "Нет данных для оценки"}
    
    result = evaluate_quality(input_text, output_text)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE calls 
        SET quality_score = ?, is_evaluated = 1 
        WHERE id = ?
    """, (result["score"], call_id))
    
    cursor.execute("""
        INSERT INTO call_details (call_id, judge_reasoning)
        VALUES (?, ?)
        ON CONFLICT(call_id) DO UPDATE SET judge_reasoning = excluded.judge_reasoning
    """, (call_id, result["reasoning"]))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Вызов {call_id} оценён: {result['score']}/5")
    return result