"""
observability/alerts.py — система алертинга через Email.
Проверяет пороговые значения метрик и отправляет уведомления.
Использует механизм cooldown, чтобы не спамить одинаковыми алертами.
"""

import os
import smtplib
import logging
import numpy as np
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

from observability.db import get_connection

load_dotenv()

logger = logging.getLogger(__name__)

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SMTP_LOGIN = os.getenv("SMTP_LOGIN")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")

# Настройки порогов алертов
ALERT_COOLDOWN_MINUTES = 30
ERROR_RATE_THRESHOLD = 0.10
ERROR_RATE_WINDOW = 20
LATENCY_P95_MULTIPLIER = 2.0
QUALITY_THRESHOLD = 3.0
QUALITY_WINDOW = 10


def send_email_alert(subject: str, message: str) -> bool:
    """Отправляет сообщение на указанный email через SMTP."""
    if not all([SMTP_LOGIN, SMTP_PASSWORD, ALERT_EMAIL]):
        logger.warning("SMTP credentials not configured. Alert skipped.")
        return False
    
    msg = MIMEMultipart()
    msg['From'] = SMTP_LOGIN
    msg['TO'] = ALERT_EMAIL
    msg['Subject'] = f"[LLM Observability] {subject}"
    
    # Форматируем сообщение для лучшей читаемости
    body = f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{message}"
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        # Используем SMTP_SSL для 465 порта
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.login(SMTP_LOGIN, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info(f"Email alert sent successfully to {ALERT_EMAIL}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
        return False


def _check_cooldown(alert_type: str) -> bool:
    """Проверяет, истек ли период ожидания (cooldown) для данного типа алерта."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts_history (
            alert_type TEXT PRIMARY KEY,
            last_triggered_at REAL,
            message TEXT
        )
    """)
    
    cursor.execute(
        "SELECT last_triggered_at FROM alerts_history WHERE alert_type = ?",
        (alert_type,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return True
    
    last_triggered = row["last_triggered_at"]
    cooldown_seconds = ALERT_COOLDOWN_MINUTES * 60
    time_since_last = datetime.now().timestamp() - last_triggered
    
    return time_since_last > cooldown_seconds


def _update_alert_history(alert_type: str, message: str):
    """Обновляет время последнего срабатывания алерта в БД."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO alerts_history (alert_type, last_triggered_at, message)
        VALUES (?, ?, ?)
        ON CONFLICT(alert_type) DO UPDATE SET 
            last_triggered_at = excluded.last_triggered_at,
            message = excluded.message
    """, (alert_type, datetime.now().timestamp(), message))
    
    conn.commit()
    conn.close()


def check_error_rate_alert() -> bool:
    """Проверяет: error rate выше порога за последние N запросов."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT success FROM calls 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (ERROR_RATE_WINDOW,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if len(rows) < ERROR_RATE_WINDOW:
        return False
    
    errors = sum(1 for row in rows if row["success"] == 0)
    error_rate = errors / len(rows)
    
    if error_rate > ERROR_RATE_THRESHOLD:
        alert_type = "high_error_rate"
        if _check_cooldown(alert_type):
            message = (
                f"Высокий уровень ошибок (Error Rate)\n"
                f"Текущий: {error_rate:.1%}\n"
                f"Порог: {ERROR_RATE_THRESHOLD:.0%}\n"
                f"Ошибок: {errors} из {len(rows)} последних запросов."
            )
            send_email_alert("HIGH ERROR RATE", message)
            _update_alert_history(alert_type, message)
            return True
    
    return False


def check_latency_p95_alert() -> bool:
    """Проверяет: p95 латентность за последний час выросла в 2 раза по сравнению с предыдущим."""
    conn = get_connection()
    cursor = conn.cursor()
    
    now = datetime.now().timestamp()
    one_hour_ago = now - 3600
    two_hours_ago = now - 7200
    
    cursor.execute("""
        SELECT latency_ms FROM calls 
        WHERE timestamp > ? AND timestamp <= ? AND success = 1
    """, (one_hour_ago, now))
    recent_latencies = [row["latency_ms"] for row in cursor.fetchall()]
    
    cursor.execute("""
        SELECT latency_ms FROM calls 
        WHERE timestamp > ? AND timestamp <= ? AND success = 1
    """, (two_hours_ago, one_hour_ago))
    previous_latencies = [row["latency_ms"] for row in cursor.fetchall()]
    
    conn.close()
    
    if len(recent_latencies) < 5 or len(previous_latencies) < 5:
        return False
    
    p95_recent = np.percentile(recent_latencies, 95)
    p95_previous = np.percentile(previous_latencies, 95)
    
    if p95_previous == 0:
        return False
    
    ratio = p95_recent / p95_previous
    
    if ratio > LATENCY_P95_MULTIPLIER:
        alert_type = "high_latency_p95"
        if _check_cooldown(alert_type):
            message = (
                f"Резкий рост латентности (P95)\n"
                f"P95 (последний час): {p95_recent:.0f} мс\n"
                f"P95 (предыдущий час): {p95_previous:.0f} мс\n"
                f"Рост в {ratio:.1f} раз (порог: {LATENCY_P95_MULTIPLIER}x)"
            )
            send_email_alert("HIGH LATENCY P95", message)
            _update_alert_history(alert_type, message)
            return True
    
    return False


def check_quality_alert() -> bool:
    """Проверяет: средняя оценка качества ниже порога за последние N оцененных вызовов."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT quality_score FROM calls 
        WHERE is_evaluated = 1 AND quality_score IS NOT NULL
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (QUALITY_WINDOW,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if len(rows) < 3:
        return False
    
    scores = [row["quality_score"] for row in rows]
    avg_score = np.mean(scores)
    
    if avg_score < QUALITY_THRESHOLD:
        alert_type = "low_quality"
        if _check_cooldown(alert_type):
            message = (
                f"Низкое качество ответов (LLM Judge)\n"
                f"Средняя оценка (последние {len(scores)}): {avg_score:.2f}\n"
                f"Порог: {QUALITY_THRESHOLD}\n"
                f"Последние оценки: {scores}"
            )
            send_email_alert("LOW QUALITY SCORE", message)
            _update_alert_history(alert_type, message)
            return True
    
    return False


def run_all_alerts():
    """Запускает все проверки алертов."""
    logger.info("Running alert checks...")
    
    triggered = []
    if check_error_rate_alert():
        triggered.append("error_rate")
    if check_latency_p95_alert():
        triggered.append("latency_p95")
    if check_quality_alert():
        triggered.append("quality")
    
    if triggered:
        logger.info(f"Alerts triggered: {triggered}")
    else:
        logger.info("No alerts triggered")
    
    return triggered