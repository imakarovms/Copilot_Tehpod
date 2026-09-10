import time
import asyncio
import functools
import inspect
import logging
from typing import Optional, Callable, Any

from observability.db import save_call, save_call_details

logger = logging.getLogger(__name__)

_config = {
    "token_counter": None,
    "default_tokenizer": None,
}


def configure(token_counter: Optional[Callable] = None, default_tokenizer: Optional[Any] = None):
    _config["token_counter"] = token_counter
    _config["default_tokenizer"] = default_tokenizer


def _extract_input_text(args: tuple, kwargs: dict) -> str:
    """
    Надежно извлекает входной текст, игнорируя 'self' и другие объекты.
    Работает как с позиционными (generate(self, query, ...)), так и с именованными аргументами.
    """
    # 1. Ищем в именованных аргументах
    for key in ["query", "prompt", "messages", "input_text", "text"]:
        if key in kwargs:
            val = kwargs[key]
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, list):
                return str(val)
    
    # 2. Ищем в позиционных аргументах. 
    # isinstance(arg, str) автоматически отсекает 'self' (экземпляр класса)
    for arg in args:
        if isinstance(arg, str) and arg.strip():
            return arg
        if isinstance(arg, list) and len(arg) > 0:
            return str(arg)
            
    return ""


def _extract_tokens_from_result(result) -> tuple[Optional[int], Optional[int]]:
    if result is None:
        return None, None
    
    if isinstance(result, dict):
        if "_obs_tokens_in" in result and "_obs_tokens_out" in result:
            return int(result["_obs_tokens_in"]), int(result["_obs_tokens_out"])
        
        if "prompt_eval_count" in result and "eval_count" in result:
            return int(result["prompt_eval_count"]), int(result["eval_count"])
            
        usage = result.get("usage")
        if isinstance(usage, dict):
            in_t = usage.get("prompt_tokens") or usage.get("input_tokens")
            out_t = usage.get("completion_tokens") or usage.get("output_tokens")
            if in_t is not None and out_t is not None:
                return int(in_t), int(out_t)
    
    usage = getattr(result, "usage", None)
    if usage is not None:
        in_t = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        out_t = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        if in_t is not None and out_t is not None:
            return int(in_t), int(out_t)
            
    return None, None


def _count_with_default_tokenizer(text: str) -> int:
    if not text:
        return 0
    
    tokenizer = _config.get("default_tokenizer")
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "encode"):
                return len(tokenizer.encode(text))
            if hasattr(tokenizer, "tokenize"):
                return len(tokenizer.tokenize(text))
            if callable(tokenizer):
                return len(tokenizer(text))
        except Exception as e:
            logger.warning(f"Default tokenizer failed: {e}")
    
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except ImportError:
        return 0
    except Exception as e:
        logger.warning(f"tiktoken failed: {e}")
        return 0


def _count_tokens(input_text: str, output_text: str, result) -> tuple[int, int]:
    custom_counter = _config.get("token_counter")
    if custom_counter is not None:
        try:
            return custom_counter(input_text, output_text, result)
        except Exception as e:
            logger.warning(f"Custom token counter failed: {e}")
    
    tokens_in, tokens_out = _extract_tokens_from_result(result)
    if tokens_in is not None and tokens_out is not None:
        return tokens_in, tokens_out
    
    tokens_in = _count_with_default_tokenizer(input_text) if input_text else 0
    tokens_out = _count_with_default_tokenizer(output_text) if output_text else 0
    return tokens_in, tokens_out


def _extract_text_from_result(result) -> str:
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
            
            # Прямое и надежное извлечение
            input_text = _extract_input_text(args, kwargs)
            output_text = _extract_text_from_result(result) if success else ""
            
            tokens_in, tokens_out = _count_tokens(input_text, output_text, result)
            model_name = _extract_model_name(result, None)
            
            try:
                call_id = save_call(
                    model_name=model_name,
                    latency_ms=latency_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    success=success,
                    error_type=error_type,
                )

                if input_text or output_text:
                    save_call_details(call_id, input_text, output_text)
                    
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
            
            # Прямое и надежное извлечение
            input_text = _extract_input_text(args, kwargs)
            output_text = _extract_text_from_result(result) if success else ""
            
            tokens_in, tokens_out = _count_tokens(input_text, output_text, result)
            model_name = _extract_model_name(result, None)
            
            try:
                call_id = save_call(
                    model_name=model_name,
                    latency_ms=latency_ms,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    success=success,
                    error_type=error_type,
                )

                if input_text or output_text:
                    save_call_details(call_id, input_text, output_text)

            except Exception as db_error:
                logger.error(f"Failed to save call metrics to DB: {db_error}")
    
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper