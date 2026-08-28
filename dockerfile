FROM python:3.12-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Скачиваем модели (если нужно)
# RUN python scripts/download_model.py && python scripts/download_llm.py

# Открываем порт
EXPOSE 8000

# Запускаем API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]