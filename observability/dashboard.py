"""
observability/dashboard.py — Streamlit-дашборд для мониторинга LLM-вызовов.

Запуск:
    streamlit run observability/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import os

# Путь к БД
DB_PATH = os.path.join(os.path.dirname(__file__), "metrics.db")


def get_connection():
    """Подключение к SQLite с WAL-режимом."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def load_calls(time_range_hours: int = 24) -> pd.DataFrame:
    """Загружает данные о вызовах за указанный период."""
    conn = get_connection()
    cutoff = datetime.now().timestamp() - (time_range_hours * 3600)
    
    query = """
        SELECT 
            id, timestamp, model_name, latency_ms, tokens_in, tokens_out,
            shadow_cost_usd, success, error_type, is_evaluated, quality_score
        FROM calls
        WHERE timestamp > ?
        ORDER BY timestamp DESC
    """
    
    df = pd.read_sql_query(query, conn, params=[cutoff])
    conn.close()
    
    # Преобразуем timestamp в datetime для удобства
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['datetime'] = pd.Series(dtype='datetime64[ns]')
    
    return df


def load_call_details(call_id: str) -> dict:
    """Загружает детали конкретного вызова (тексты промпта и ответа)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT input_text, output_text, judge_reasoning FROM call_details WHERE call_id = ?",
        (call_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "input_text": row["input_text"],
            "output_text": row["output_text"],
            "judge_reasoning": row["judge_reasoning"],
        }
    return {}


def calculate_p95(group: pd.Series) -> float:
    """Вычисляет 95-й перцентиль."""
    return np.percentile(group.dropna(), 95) if len(group) > 0 else 0


def calculate_p50(group: pd.Series) -> float:
    """Вычисляет 50-й перцентиль (медиана)."""
    return np.percentile(group.dropna(), 50) if len(group) > 0 else 0


# ─────────────────────────────────────────────────────────────
# UI: Настройка страницы
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Observability Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 LLM Observability Dashboard")
st.markdown("Мониторинг латентности, стоимости и качества вызовов LLM")

# ─────────────────────────────────────────────────────────────
# Сайдбар: Фильтры
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Настройки")

time_range = st.sidebar.slider(
    "Период (часов)",
    min_value=1,
    max_value=168,  # 7 дней
    value=24,
    step=1,
)

show_details = st.sidebar.checkbox("Показывать детали вызовов", value=False)

# ─────────────────────────────────────────────────────────────
# Загрузка данных
# ────────────────────────────────────────────────────────────
df = load_calls(time_range)

if df.empty:
    st.warning(f"Нет данных за последние {time_range} часов. Сделайте хотя бы один запрос к LLM.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# KPI: Общая статистика
# ─────────────────────────────────────────────────────────────
st.subheader("📈 Общая статистика")

col1, col2, col3, col4, col5 = st.columns(5)

total_calls = len(df)
total_tokens_in = df["tokens_in"].sum()
total_tokens_out = df["tokens_out"].sum()
total_cost = df["shadow_cost_usd"].sum()
error_rate = (df["success"] == 0).mean() * 100

col1.metric("Всего вызовов", total_calls)
col2.metric("Входные токены", f"{total_tokens_in:,}")
col3.metric("Выходные токены", f"{total_tokens_out:,}")
col4.metric("Теневая стоимость", f"${total_cost:.4f}")
col5.metric("Error Rate", f"{error_rate:.1f}%")

st.markdown("---")

# ────────────────────────────────────────────────────────────
# График 1: Латентность (p50/p95) по времени
# ─────────────────────────────────────────────────────────────
st.subheader("️ Латентность (p50/p95)")

# Группируем по часам
df_hourly = df.set_index("datetime").resample("1h").agg({
        "latency_ms": ["count", "mean", lambda x: calculate_p50(x), lambda x: calculate_p95(x)]
})
df_hourly.columns = ["count", "mean", "p50", "p95"]
df_hourly = df_hourly.reset_index()

fig_latency = go.Figure()
fig_latency.add_trace(go.Scatter(
    x=df_hourly["datetime"],
    y=df_hourly["p50"],
    mode="lines+markers",
    name="p50 (медиана)",
    line=dict(color="blue", width=2),
))
fig_latency.add_trace(go.Scatter(
    x=df_hourly["datetime"],
    y=df_hourly["p95"],
    mode="lines+markers",
    name="p95",
    line=dict(color="red", width=2),
))
fig_latency.update_layout(
    xaxis_title="Время",
    yaxis_title="Латентность (мс)",
    hovermode="x unified",
)
st.plotly_chart(fig_latency, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# График 2: Объём запросов
# ─────────────────────────────────────────────────────────────
st.subheader("📊 Объём запросов")

fig_volume = px.bar(
    df_hourly,
    x="datetime",
    y="count",
    labels={"datetime": "Время", "count": "Количество вызовов"},
    title="Запросы в час",
)
st.plotly_chart(fig_volume, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# График 3: Error Rate
# ─────────────────────────────────────────────────────────────
st.subheader("❌ Error Rate")

df_errors = df.set_index("datetime").resample("1h").agg({
    "success": ["count", lambda x: (x == 0).sum()]
})
df_errors.columns = ["total", "errors"]
df_errors["error_rate"] = (df_errors["errors"] / df_errors["total"]) * 100
df_errors = df_errors.reset_index()

fig_errors = px.line(
    df_errors,
    x="datetime",
    y="error_rate",
    labels={"datetime": "Время", "error_rate": "Error Rate (%)"},
    title="Процент ошибок по часам",
    markers=True,
)
fig_errors.update_traces(line_color="red", marker_size=8)
st.plotly_chart(fig_errors, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# График 4: Накопленная теневая экономия
# ─────────────────────────────────────────────────────────────
st.subheader("💰 Накопленная теневая экономия")

df["cumulative_cost"] = df["shadow_cost_usd"].cumsum()

fig_cost = px.area(
    df.sort_values("timestamp"),
    x="datetime",
    y="cumulative_cost",
    labels={"datetime": "Время", "cumulative_cost": "Накопленная стоимость (USD)"},
    title="Сколько бы вы заплатили, используя API",
)
fig_cost.update_traces(line_color="green")
st.plotly_chart(fig_cost, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# График 5: Тренд качества (если есть оценки)
# ─────────────────────────────────────────────────────────────
df_evaluated = df[df["is_evaluated"] == 1]

if not df_evaluated.empty:
    st.subheader("⭐ Тренд качества")
    
    df_quality = df_evaluated.set_index("datetime").resample("1h").agg({
        "quality_score": ["count", "mean"]
    })
    df_quality.columns = ["count", "avg_score"]
    df_quality = df_quality.reset_index()
    
    fig_quality = px.line(
        df_quality,
        x="datetime",
        y="avg_score",
        labels={"datetime": "Время", "avg_score": "Средняя оценка"},
        title="Средняя оценка качества по часам",
        markers=True,
    )
    fig_quality.update_traces(line_color="purple", marker_size=8)
    fig_quality.update_yaxes(range=[0, 5])
    st.plotly_chart(fig_quality, use_container_width=True)
else:
    st.info("Оценки качества ещё не собраны. Независимый судья запустится после накопления данных.")

# ─────────────────────────────────────────────────────────────
# Таблица последних вызовов
# ─────────────────────────────────────────────────────────────
st.subheader(" Последние вызовы")

df_display = df[["datetime", "model_name", "latency_ms", "tokens_in", "tokens_out", "shadow_cost_usd", "success"]].head(20)
df_display["success"] = df_display["success"].map({1: "✅ OK", 0: "❌ Error"})

st.dataframe(df_display, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Детали конкретного вызова
# ─────────────────────────────────────────────────────────────
if show_details:
    st.subheader("🔍 Детали вызова")
    
    call_id = st.selectbox(
        "Выберите вызов",
        options=df["id"].tolist(),
        format_func=lambda x: f"{x[:8]}... ({df[df['id']==x]['datetime'].values[0]})",
    )
    
    if call_id:
        details = load_call_details(call_id)
        
        if details:
            st.text_area("Промпт", details["input_text"], height=200, disabled=True)
            st.text_area("Ответ модели", details["output_text"], height=200, disabled=True)
            
            if details["judge_reasoning"]:
                st.text_area("Рассуждение судьи", details["judge_reasoning"], height=150, disabled=True)
        else:
            st.warning("Детали для этого вызова не найдены.")