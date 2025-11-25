import streamlit as st
import database
import pandas as pd
import numpy as np
import time

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
LOW_DQR_THRESHOLD = 0.5
LOW_CONF_THRESHOLD = 0.5

st.set_page_config(
    page_title="MediCure ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 MediCure ML Intelligence Dashboard")
st.markdown("Monitor AI performance, OCR quality, system load, and medicine insights.")

# ---------------------------------------------------------------
# LOAD METRICS FROM DATABASE
# ---------------------------------------------------------------

def load_metrics():
    """Loads scan metrics from DB and returns a metrics dictionary."""
    raw = database.get_all_scan_metrics()

    if not raw:
        return {"df": pd.DataFrame(), "total": 0}

    df = pd.DataFrame(raw)

    # Normalize columns
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["dqr"] = pd.to_numeric(df["dqr"], errors="coerce").fillna(0.0)
    df["ocr_text_length"] = pd.to_numeric(df.get("ocr_text_length", 0), errors="coerce").fillna(0).astype(int)

    total = len(df)

    if total == 0:
        return {"df": df, "total": 0}

    # ---------------------------------------------------------------
    # METRIC COMPUTATIONS
    # ---------------------------------------------------------------

    metrics = {}

    # Performance
    metrics["avg_duration"] = df["duration"].mean()
    metrics["max_duration"] = df["duration"].max()
    metrics["duration_std"] = df["duration"].std()

    # Confidence
    metrics["avg_conf"] = df["confidence"].mean() * 100
    metrics["high_conf_rate"] = (df["confidence"] >= 0.8).mean() * 100
    metrics["low_conf_count"] = (df["confidence"] < LOW_CONF_THRESHOLD).sum()
    metrics["low_conf_rate"] = metrics["low_conf_count"] / total * 100

    # Unknown / failure
    metrics["unknown_count"] = df["is_unknown"].sum()
    metrics["unknown_rate"] = metrics["unknown_count"] / total * 100

    # DQR
    metrics["avg_dqr"] = df["dqr"].mean()
    metrics["low_dqr_count"] = (df["dqr"] < LOW_DQR_THRESHOLD).sum()
    metrics["low_dqr_rate"] = metrics["low_dqr_count"] / total * 100

    if metrics["unknown_count"] > 0:
        correlation = df[(df["is_unknown"]) & (df["dqr"] < LOW_DQR_THRESHOLD)].shape[0]
        metrics["correlation_rate"] = correlation / metrics["unknown_count"] * 100
    else:
        metrics["correlation_rate"] = 0.0

    # OCR
    metrics["ocr_text_len_mean"] = df["ocr_text_length"].mean()
    metrics["ocr_fail_count"] = (df["ocr_text_length"] < 5).sum()
    metrics["ocr_fail_rate"] = metrics["ocr_fail_count"] / total * 100

    metrics["matching_score"] = (df["confidence"] * df["dqr"]).mean()

    # Parsing / Alternatives
    df["parsing_success"] = df["ai_parsed"].apply(lambda x: isinstance(x, dict) and len(x) > 0)
    metrics["parsing_success_rate"] = df["parsing_success"].mean() * 100

    df["alt_success"] = df["ai_parsed"].apply(
        lambda x: isinstance(x, dict) and len(x.get("alternatives", [])) > 0
    )
    metrics["alt_success_rate"] = df["alt_success"].mean() * 100

    # Brand / generic
    metrics["brand_rate"] = df["brand_name"].astype(bool).mean() * 100
    metrics["generic_rate"] = df["generic_name"].astype(bool).mean() * 100

    # Scans per minute
    if df["timestamp"].notna().any():
        df["minute"] = df["timestamp"].dt.floor("T")
        scans = df.groupby("minute").size()
        metrics["avg_scans_per_min"] = scans.mean()
        metrics["peak_load"] = scans.max()
    else:
        metrics["avg_scans_per_min"] = 0
        metrics["peak_load"] = 0

    # Repeat scan rate
    counts = df["medicine_name"].value_counts()
    metrics["repeat_rate"] = counts[counts > 1].sum() / total * 100

    # Top lists
    metrics["top_meds"] = counts.head(8)

    alt_list = df["ai_parsed"].apply(
        lambda x: x.get("alternatives") if isinstance(x, dict) else []
    ).explode()
    metrics["top_alts"] = alt_list.value_counts().head(8)

    # OCR → Medicine match
    metrics["ocr_match_rate"] = df["ocr_contains_name"].mean() * 100

    # Reliability score
    metrics["reliability"] = (
        df["confidence"].mean()
        * (1 - metrics["low_dqr_rate"] / 100)
        * (1 - metrics["unknown_rate"] / 100)
    ) * 100

    # Consistency
    metrics["consistency"] = df.groupby("medicine_name")["confidence"].std().mean()

    metrics["df"] = df
    metrics["total"] = total

    return metrics


# ---------------------------------------------------------------
# LOAD DATA (SAFE)
# ---------------------------------------------------------------
try:
    m = load_metrics()
except Exception as e:
    st.error(f"Failed to load metrics: {e}")
    m = {"df": pd.DataFrame(), "total": 0}

df = m["df"]
total = m["total"]

if total == 0:
    st.info("No scan data available yet. Perform scans to see analytics.")
    st.stop()

# ---------------------------------------------------------------
# UI SECTIONS
# ---------------------------------------------------------------

st.markdown("---")
st.header("🚀 System Performance Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Scans", f"{total:,}")
col2.metric("Avg. Duration", f"{m['avg_duration']:.2f}s", delta=f"{m['max_duration']:.2f}s max")
col3.metric("AI Confidence", f"{m['avg_conf']:.1f}%")
col4.metric("Avg DQR", f"{m['avg_dqr']:.2f}", delta=f"Threshold {LOW_DQR_THRESHOLD}")

# ---------------------------------------------------------------

st.markdown("---")
st.header("⚠️ Quality & Failure Analysis")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Unknown Medicines", f"{m['unknown_count']:,}", delta=f"{m['unknown_rate']:.1f}%")
col2.metric("Low Confidence (<50%)", f"{m['low_conf_count']:,}", delta=f"{m['low_conf_rate']:.1f}%")
col3.metric("Poor Image Quality", f"{m['low_dqr_count']:,}", delta=f"{m['low_dqr_rate']:.1f}%")
col4.metric("Low DQR → Unknown", f"{m['correlation_rate']:.1f}%")

# ---------------------------------------------------------------

st.markdown("---")
st.header("🧾 OCR & AI Parsing Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg OCR Length", f"{m['ocr_text_len_mean']:.0f} chars")
col2.metric("OCR Failures", f"{m['ocr_fail_count']:,}", delta=f"{m['ocr_fail_rate']:.1f}%")
col3.metric("Parsing Success", f"{m['parsing_success_rate']:.1f}%")
col4.metric("Alternative Success", f"{m['alt_success_rate']:.1f}%")

# ---------------------------------------------------------------

st.markdown("---")
st.header("⚙️ System Load Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Scans / Min", f"{m['avg_scans_per_min']:.2f}")
col2.metric("Peak Load", f"{m['peak_load']}")
col3.metric("Latency Std Dev", f"{m['duration_std']:.3f}s")
col4.metric("AI Reliability Score", f"{m['reliability']:.1f}%")

# ---------------------------------------------------------------

st.markdown("---")
st.header("💊 Medicine Insights")

left, right = st.columns([1.5, 1])

with left:
    st.subheader("Top Scanned Medicines")
    st.table(m["top_meds"].rename_axis("Medicine").reset_index(name="Count"))

    st.subheader("Most Suggested Alternatives")
    st.table(m["top_alts"].rename_axis("Alternative").reset_index(name="Count"))

with right:
    st.subheader("Additional Metrics")
    st.metric("Repeat Scan Rate", f"{m['repeat_rate']:.1f}%")
    st.metric("Brand Detection Rate", f"{m['brand_rate']:.1f}%")
    st.metric("Generic Detection Rate", f"{m['generic_rate']:.1f}%")
    st.metric("OCR → Name Match", f"{m['ocr_match_rate']:.1f}%")

# ---------------------------------------------------------------

st.markdown("---")
st.header("📈 Unknown Medicine Trends")

unknown_df = df[df["is_unknown"]]

if not unknown_df.empty:
    ts = (
        unknown_df.groupby(unknown_df["timestamp"].dt.date)
        .size()
        .reset_index(name="Count")
    )
    ts.rename(columns={"timestamp": "Date"}, inplace=True)

    st.area_chart(ts, x="Date", y="Count")
else:
    st.info("No unknown medicine scans detected yet.")

# ---------------------------------------------------------------

st.markdown("---")
st.header("📋 Recently Processed Scans")

display_cols = [
    "timestamp", "medicine_name", "duration", "confidence", "dqr",
    "ocr_text_length", "is_unknown", "is_low_confidence",
    "parsing_success", "alt_success"
]

if all(c in df.columns for c in display_cols):
    table = df[display_cols].copy()
    table.rename(columns={
        "timestamp": "Time",
        "medicine_name": "Medicine",
        "duration": "Duration (s)",
        "confidence": "Confidence",
        "dqr": "DQR",
        "ocr_text_length": "OCR Len",
        "is_unknown": "Unknown",
        "is_low_confidence": "Low Conf",
        "parsing_success": "Parsed",
        "alt_success": "Alt OK"
    }, inplace=True)

    table["Time"] = table["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    table["Confidence"] = (table["Confidence"] * 100).map("{:.1f}%".format)
    table["DQR"] = table["DQR"].map("{:.2f}".format)

    st.dataframe(table, use_container_width=True, hide_index=True)

else:
    st.warning("Some fields were missing from database. Recent scans table is limited.")
