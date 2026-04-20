# app/components/metrics_dashboard.py
"""
MLOps Metrics Dashboard: Displays real-time metrics, history charts,
and system health in the Streamlit UI.
"""

import time
from typing import Dict, Any, List, Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def render_metric_cards(stats: Dict[str, Any], gpu_stats: Dict[str, Any]):
    """Render top-level KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Generations",
            f"{stats.get('total_generations', 0):,}",
            help="All-time generation count",
        )

    with col2:
        avg_lat = stats.get("latency", {}).get("mean_s", 0)
        st.metric(
            "Avg Latency",
            f"{avg_lat:.1f}s" if avg_lat else "—",
            help="Mean generation time",
        )

    with col3:
        err_rate = stats.get("error_rate", 0) * 100
        delta_color = "normal" if err_rate < 5 else "inverse"
        st.metric(
            "Error Rate",
            f"{err_rate:.1f}%",
            delta=f"{err_rate - 5:.1f}% vs 5% threshold" if err_rate > 5 else None,
            delta_color=delta_color,
        )

    with col4:
        speed = stats.get("speed", {}).get("mean_steps_per_s", 0)
        st.metric(
            "Avg Speed",
            f"{speed:.1f} steps/s" if speed else "—",
            help="Inference speed",
        )


def render_gpu_card(gpu_stats: Dict[str, Any]):
    """Render GPU memory gauge."""
    if not gpu_stats.get("available"):
        st.info("🖥️ Running on CPU — no GPU metrics available")
        return

    st.markdown(f"**GPU**: {gpu_stats.get('device_name', 'Unknown')}")

    total = gpu_stats.get("total_mb", 1)
    used = gpu_stats.get("allocated_mb", 0)
    pct = (used / total) * 100 if total > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("VRAM Used", f"{used:.0f} MB")
    with col2:
        st.metric("VRAM Total", f"{total:.0f} MB")

    color = "#4caf50" if pct < 70 else "#FF9933" if pct < 90 else "#f44336"
    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,0.05); border-radius:8px; overflow:hidden; height:12px; margin:4px 0; border: 1px solid rgba(212, 175, 55, 0.1);">
            <div style="background:{color}; width:{pct:.1f}%; height:100%; transition: width 0.5s; box-shadow: 0 0 10px {color}44;"></div>
        </div>
        <p style="color:#b0a898; font-size:0.8rem; margin-top:5px;">{pct:.1f}% VRAM utilized</p>
        """,
        unsafe_allow_html=True,
    )


def render_latency_chart(history: List[Dict[str, Any]]):
    """Plot generation latency over time."""
    if len(history) < 2:
        st.info("Not enough data yet — generate some images!")
        return

    times = [h["generation_time_s"] for h in reversed(history)]
    ids = [h["request_id"] for h in reversed(history)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(times))),
        y=times,
        mode="lines+markers",
        name="Latency (s)",
        line=dict(color="#e8a87c", width=2),
        marker=dict(size=6, color="#e8a87c"),
        hovertemplate="<b>%{y:.1f}s</b><br>Request: %{customdata}<extra></extra>",
        customdata=ids,
    ))

    # Add rolling average
    if len(times) >= 3:
        window = min(5, len(times))
        rolling_avg = [
            sum(times[max(0, i - window):i + 1]) / len(times[max(0, i - window):i + 1])
            for i in range(len(times))
        ]
        fig.add_trace(go.Scatter(
            x=list(range(len(rolling_avg))),
            y=rolling_avg,
            mode="lines",
            name="Rolling Avg",
            line=dict(color="#7ecce8", width=2, dash="dot"),
        ))

    fig.update_layout(
        title="Generation Latency",
        xaxis_title="Generation #",
        yaxis_title="Time (s)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8F1E9'),
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_scheduler_pie(stats: Dict[str, Any]):
    """Pie chart of scheduler usage."""
    scheduler_data = stats.get("scheduler_usage", {})
    if not scheduler_data:
        return

    fig = go.Figure(go.Pie(
        labels=list(scheduler_data.keys()),
        values=list(scheduler_data.values()),
        hole=0.45,
        textinfo="label+percent",
        marker=dict(colors=["#5D101D", "#FF9933", "#D4AF37", "#005B96", "#8B1A2C"]),
    ))
    fig.update_layout(
        title="Scheduler Usage",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8F1E9'),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_latency_histogram(history: List[Dict[str, Any]]):
    """Distribution of generation times."""
    if len(history) < 5:
        return

    times = [h["generation_time_s"] for h in history if h.get("success", True)]

    fig = go.Figure(go.Histogram(
        x=times,
        nbinsx=15,
        marker_color="#FF9933",
        opacity=0.8,
    ))
    fig.update_layout(
        title="Latency Distribution",
        xaxis_title="Time (s)",
        yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8F1E9'),
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_generation_history_table(history: List[Dict[str, Any]]):
    """Render recent generations as a styled table."""
    if not history:
        st.info("No generation history yet.")
        return

    import pandas as pd

    rows = []
    for h in history[:20]:
        rows.append({
            "ID": h.get("request_id", "?"),
            "Time (s)": f"{h.get('generation_time_s', 0):.1f}",
            "Steps/s": f"{h.get('steps_per_second', 0):.1f}",
            "Steps": h.get("steps", 0),
            "Size": f"{h.get('width', 0)}×{h.get('height', 0)}",
            "Scheduler": h.get("scheduler", "?"),
            "Status": "✅" if h.get("success", True) else "❌",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def render_mlops_dashboard(
    stats: Dict[str, Any],
    gpu_stats: Dict[str, Any],
    history: List[Dict[str, Any]],
    mlflow_enabled: bool = False,
    mlflow_tracking_uri: str = "http://localhost:5012",
    prometheus_enabled: bool = True,
    prometheus_ui_url: str = "http://localhost:9090",
):
    """Full MLOps dashboard tab."""
    st.markdown("## 📊 MLOps Monitoring Dashboard")

    # KPI Cards
    render_metric_cards(stats, gpu_stats)
    st.divider()

    # GPU + Charts row
    col_gpu, col_sched = st.columns([1, 1])
    with col_gpu:
        st.markdown("#### GPU Status")
        render_gpu_card(gpu_stats)

    with col_sched:
        st.markdown("#### Scheduler Distribution")
        render_scheduler_pie(stats)

    # Latency charts
    st.markdown("#### Performance Trends")
    col_lat, col_hist = st.columns([2, 1])
    with col_lat:
        render_latency_chart(history)
    with col_hist:
        render_latency_histogram(history)

    # History table
    st.markdown("#### Recent Generations")
    render_generation_history_table(history)

    # External Tools Section
    st.divider()
    st.markdown("#### 🔗 External Monitoring Tools")
    tool_col1, tool_col2 = st.columns(2)
    
    with tool_col1:
        if mlflow_enabled:
            st.markdown(
                f"""
                <div style='background:rgba(1,121,209,0.1); border:1px solid rgba(1,121,209,0.3); border-radius:10px; padding:15px;'>
                    <h5 style='margin:0;'>📈 MLflow Tracking</h5>
                    <p style='font-size:0.85rem; color:#888; margin:5px 0 10px 0;'>Experiment logs, artifacts & run comparison</p>
                    <a href='{mlflow_tracking_uri}' target='_blank' style='text-decoration:none;'>
                        <button style='background-color:#0179d1; color:white; border:none; padding:6px 12px; border-radius:5px; cursor:pointer;'>Open MLflow UI</button>
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("MLflow tracking is disabled")

    with tool_col2:
        if prometheus_enabled:
            st.markdown(
                f"""
                <div style='background:rgba(230,81,0,0.1); border:1px solid rgba(230,81,0,0.3); border-radius:10px; padding:15px;'>
                    <h5 style='margin:0;'>🔥 Prometheus UI</h5>
                    <p style='font-size:0.85rem; color:#888; margin:5px 0 10px 0;'>Real-time metrics, queries & expressions</p>
                    <a href='{prometheus_ui_url}' target='_blank' style='text-decoration:none;'>
                        <button style='background-color:#e65100; color:white; border:none; padding:6px 12px; border-radius:5px; cursor:pointer;'>Open Prometheus</button>
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Prometheus metrics are disabled")

    # Latency percentiles
    latency = stats.get("latency", {})
    if latency.get("mean_s", 0) > 0:
        st.divider()
        st.markdown("#### Latency Percentiles")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.metric("P50 (Median)", f"{latency.get('median_s', 0):.1f}s")
        with p_col2:
            st.metric("Mean", f"{latency.get('mean_s', 0):.1f}s")
        with p_col3:
            st.metric("P95", f"{latency.get('p95_s', 0):.1f}s")
        with p_col4:
            st.metric("P99", f"{latency.get('p99_s', 0):.1f}s")
