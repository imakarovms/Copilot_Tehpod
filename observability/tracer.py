import time
import asyncio
import functools
import inspect
import logging
from typing import Optional, Callable, Any

from observability.db import save_call

logger = logging.getLogger(__name__)


# Глобальная конфигурация библиотеки.
# Задается один раз при старте приложения, например:
#   from observability.tracer import configure
#   configure(token_counter=my_custom_counter)
_config = {
    "token_counter": None,  # Пользовательская функция подсчета токенов
    "default_tokenizer": None,  # Токенайзер для фоллбэка (например, tiktoken encoder)
}


def configure(token_counter: Optional[Callable] = None, default_tokenizer: Optional[Any] = None):
    """
    Настраивает библиотеку observability.
    
    token_counter — функция (input_text, output_text, result) -> (tokens_in, tokens_out).
        Используется, если у вас специфичный бэкенд и вы хотите полностью контролировать подсчет.
    
    default_tokenizer — объект токенайзера, у которого есть метод encode(text) -> list[int].
        Например, tiktoken.get_encoding("cl100k_base") или AutoTokenizer.from_pretrained(...).
    """
    _config["token_counter"] = token_counter
    _config["default_tokenizer"] = default_tokenizer


def _extract_tokens_from_result(result) -> tuple[Optional[int], Optional[int]]:
    """
    Уровень 1: извлекает количество токенов из ответа.
    Поддерживает сырой ответ Ollama/OpenAI или наш кастомный словарь с метриками.
    """
    if result is None:
        return None, None
    
    if isinstance(result, dict):
        # Вариант А: Кастомные ключи, добавленные в generator.py
        if "_obs_tokens_in" in result and "_obs_tokens_out" in result:
            return int(result["_obs_tokens_in"]), int(result["_obs_tokens_out"])
        
        # Вариант Б: Сырой ответ Ollama API
        if "prompt_eval_count" in result and "eval_count" in result:
            return int(result["prompt_eval_count"]), int(result["eval_count"])
            
        # Вариант В: OpenAI / Anthropic формат
        usage = result.get("usage")
        if isinstance(usage, dict):
            in_t = usage.get("prompt_tokens") or usage.get("input_tokens")
            out_t = usage.get("completion_tokens") or usage.get("output_tokens")
            if in_t is not None and out_t is not None:
                return int(in_t), int(out_t)
    
    # Вариант Г: Объект с вложенным атрибутом usage (OpenAI SDK)
    usage = getattr(result, "usage", None)
    if usage is not None:
        in_t = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        out_t = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        if in_t is not None and out_t is not None:
            return int(in_t), int(out_t)
            
    return None, None


def _count_with_default_tokenizer(text: str) -> int:
    """
    Уровень 2/3: считает токены через настроенный токенайзер или tiktoken.
    """
    if not text:
        return 0
    
    # Если пользователь задал свой токенайзер через configure()
    tokenizer = _config.get("default_tokenizer")
    if tokenizer is not None:
        try:
            # Поддержка разных интерфейсов: encode(), tokenize(), __call__()
            if hasattr(tokenizer, "encode"):
                return len(tokenizer.encode(text))
            if hasattr(tokenizer, "tokenize"):
                return len(tokenizer.tokenize(text))
            if callable(tokenizer):
                return len(tokenizer(text))
        except Exception as e:
            logger.warning(f"Default tokenizer failed: {e}")
    
    # Фоллбэк на tiktoken, если установлен
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except ImportError:
        logger.warning("tiktoken not installed, cannot count tokens accurately")
        return 0
    except Exception as e:
        logger.warning(f"tiktoken failed: {e}")
        return 0


def _count_tokens(input_text: str, output_text: str, result) -> tuple[int, int]:
    """
    Главная функция подсчета токенов. Применяет трехуровневую стратегию.
    """
    # Если пользователь задал полностью кастомный счетчик — используем его
    custom_counter = _config.get("token_counter")
    if custom_counter is not None:
        try:
            return custom_counter(input_text, output_text, result)
        except Exception as e:
            logger.warning(f"Custom token counter failed: {e}")
    
    # Уровень 1: пробуем вытащить из результата
    tokens_in, tokens_out = _extract_tokens_from_result(result)
    if tokens_in is not None and tokens_out is not None:
        return tokens_in, tokens_out
    
    # Уровень 2/3: считаем через токенайзер
    tokens_in = _count_with_default_tokenizer(input_text) if input_text else 0
    tokens_out = _count_with_default_tokenizer(output_text) if output_text else 0
    return tokens_in, tokens_out


def _extract_model_from_result(result):
    """Извлекает объект модели из результата, если он есть."""
    if hasattr(result, "model"):
        return result.model
    if hasattr(result, "llm"):
        return result.llm
    if isinstance(result, dict) and "model" in result:
        return result["model"]
    return None


def _extract_text_from_result(result) -> str:
    """Извлекает текстовый ответ из результата генерации."""
    if hasattr(result, "text"):
        return result.text
    if hasattr(result, "answer"):
        return result.answer
    if hasattr(result, "content"):
        return result.content
    if isinstance(result, dict):
        return result.get("answer") or result.get("text") or result.get("content") or ""
    if isinstance(result, str):
        return result
    return ""


def _extract_model_name(result, model) -> str:
    """Извлекает имя модели для записи в БД."""
    if isinstance(result, dict) and "_obs_model" in result:
        return str(result["_obs_model"])
    if model is not None:
        name = getattr(model, "model_name", None) or getattr(model, "model_path", None)
        if name:
            return str(name)
    if isinstance(result, dict):
        return str(result.get("model", "unknown"))
    return "unknown"


def observe(func):
    """
    Декоратор для инструментирования вызовов LLM.
    
    Не знает и не должен знать, какой бэкенд используется (llama-cpp, Ollama, OpenAI, vLLM).
    Подсчет токенов происходит через трехуровневую стратегию:
      1. Извлечение из ответа бэкенда (если бэкенд вернул usage)
      2. Подсчет через настроенный токенайзер
      3. Фоллбэк на tiktoken
    
    Поддерживает как sync, так и async функции.
    """
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        success = 1
        error_type = None
        result = None
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            success = 0
            error_type = type(e).__name__
            logger.error(f"LLM call failed: {e}")
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            model = _extract_model_from_result(result)
            input_text = kwargs.get("prompt") or kwargs.get("query") or kwargs.get("messages") or ""
            # Если messages — это список (OpenAI-формат), сериализуем для подсчета токенов
            if isinstance(input_text, list):
                input_text = str(input_text)
            
            output_text = _extract_text_from_result(result) if success else ""
            tokens_in, tokens_out = _count_tokens(input_text, output_text, result)
            model_name = _extract_model_name(result, model)
            
            try:
                save_call(
                    model_name=model_name,
                    latency_ms=latency_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    success=success,
                    error_type=error_type,
                )
            except Exception as db_error:
                logger.error(f"Failed to save call metrics to DB: {db_error}")
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        success = 1
        error_type = None
        result = None
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = 0
            error_type = type(e).__name__
            logger.error(f"LLM call failed: {e}")
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            model = _extract_model_from_result(result)
            input_text = kwargs.get("prompt") or kwargs.get("query") or kwargs.get("messages") or ""
            if isinstance(input_text, list):
                input_text = str(input_text)
            
            output_text = _extract_text_from_result(result) if success else ""
            tokens_in, tokens_out = _count_tokens(input_text, output_text, result)
            model_name = _extract_model_name(result, model)
            
            try:
                save_call(
                    model_name=model_name,
                    latency_ms=latency_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    success=success,
                    error_type=error_type,
                )
            except Exception as db_error:
                logger.error(f"Failed to save call metrics to DB: {db_error}")
    
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper