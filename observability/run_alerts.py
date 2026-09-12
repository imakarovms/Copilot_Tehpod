"""
observability/run_alerts.py — CLI-скрипт для проверки и отправки алертов.
"""

import sys
import os
import argparse
import logging

# Явно добавляем корень проекта в путь поиска модулей.
# Это решает проблему "No module named 'observability'" при прямом запуске скрипта.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from observability.alerts import run_all_alerts, send_email_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def send_test_alert():
    """Отправляет тестовое сообщение на email."""
    message = "Это тестовое сообщение от системы мониторинга LLM Observability.\nЕсли вы его видите, настройка SMTP работает корректно."
    success = send_email_alert("TEST ALERT", message)
    if success:
        print("✅ Тестовое сообщение успешно отправлено на email.")
    else:
        print("❌ Не удалось отправить сообщение. Проверьте SMTP настройки в .env")


def main():
    parser = argparse.ArgumentParser(description="LLM Monitoring Alerts")
    parser.add_argument("--test", action="store_true", help="Отправить тестовое сообщение")
    
    args = parser.parse_args()
    
    if args.test:
        send_test_alert()
    else:
        triggered = run_all_alerts()
        if triggered:
            print(f"🚨 Алерты сработали: {', '.join(triggered)}")
        else:
            print("✅ Все метрики в норме")


if __name__ == "__main__":
    main()