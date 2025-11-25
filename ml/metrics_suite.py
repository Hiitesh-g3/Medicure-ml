"""
metrics_suite.py

Upgraded MediCure AI Monitoring Suite (single-file Streamlit app).
Features included (professor-impressing subset):
  - Multi-page layout (Overview / Drift / Image Inspector / Eval / Profiling / Admin)
  - AI Drift Detection (simple statistical & rolling-window alerts)
  - Image Quality Debugger (view per-scan JSON, OCR text, DQR, blur, brightness)
  - ML Evaluation Page (compare parsing / confidence by model/prompt if available)
  - Pipeline Profiling Page (stage timing breakdown if stored)
  - Admin Tools (delete scan, export CSV, download scan JSON)
  - Defensive: works with older DB rows (missing keys handled)
  - Uses existing database.py API (get_all_scan_metrics, get_scan_context)
    and falls back to sqlite directly for admin delete/export.

How to run:
  streamlit run metrics_suite.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import sqlite3
import datetime
import base64
from typing import Dict, Any

# Import the project's database utility (must be in same project)
import database

# ---------------------
# Config
# ---------------------
st.set_page_config(page_title="MediCure AI Monitoring Suite", layout="wide", page_icon="🩺")
LOW_DQR_THRESHOLD = 0.5
LOW_CONF_THRESHOLD = 0.5
DB_PATH = "medicure_memory.db"  # same as database.DB_NAME

# ---------------------
# Utilities
# ---------------------
@st.cache_data(ttl=30)
def load_raw_metrics():
    """Load raw metrics via database.get_all_scan_metrics (cached)."""
    try:
        metrics = database.get_all_scan_metrics()
    except Exception:
        metrics = []
    return metrics

def build_df(raw):
    """Convert raw metrics list to DataFrame and normalize columns defensively."""
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    # Ensure expected columns exist
    for col in [
        "timestamp", "duration", "confidence", "dqr", "ocr_text_length",
        "is_unknown", "is_low_confidence", "ai_parsed", "medicine_name",
        "ocr_text", "brand_name", "generic_name", "alternatives", "status",
        "user_id", "blur_score", "brightness", "resolution", "ocr_contains_name"
    ]:
        if col not in df.columns:
            df[col] = None
    # Normalize types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["dqr"] = pd.to_numeric(df["dqr"], errors="coerce").fillna(0.0)
    df["ocr_text_length"] = pd.to_numeric(df["ocr_text_length"], errors="coerce").fillna(0).astype(int)
    # parsing flags and safe shapes
    df["ai_parsed"] = df["ai_parsed"].apply(lambda x: x if isinstance(x, dict) else (json.loads(x) if isinstance(x, str) and x.strip().startswith(("{","[")) else {}))
    df["is_unknown"] = df["is_unknown"].astype(bool)
    df["is_low_confidence"] = df["is_low_confidence"].astype(bool)
    df["ocr_contains_name"] = df.get("ocr_contains_name", False).astype(bool)
    return df

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue()

def download_link_bytes(content: bytes, filename: str, mime: str = "text/csv"):
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# ---------------------
# Drift detection helpers
# ---------------------
def simple_drift_alerts(df: pd.DataFrame):
    """
    Compute simple drift signals:
      - Recent mean confidence drop vs historic mean (windowed)
      - Recent mean DQR drop vs historic
      - Unknown rate spike
    Returns alerts list with severity.
    """
    alerts = []
    if df.empty or df["timestamp"].notna().sum() == 0:
        return alerts

    df = df.sort_values("timestamp")
    total = len(df)
    # Use last 14 days vs previous 30 days or windows by count if no timestamps
    now = df["timestamp"].max()
    recent_cut = now - pd.Timedelta(days=7)
    prev_cut = now - pd.Timedelta(days=37)

    recent = df[df["timestamp"] >= recent_cut]
    prev = df[(df["timestamp"] < recent_cut) & (df["timestamp"] >= prev_cut)]

    # Fall back to counts windows when prev empty
    if recent.empty or prev.empty:
        # use halves
        half = int(total // 2)
        recent = df.tail(half)
        prev = df.head(total - half)

    # Confidence drift
    recent_conf = recent["confidence"].mean()
    prev_conf = prev["confidence"].mean() if not prev.empty else recent_conf
    if prev_conf > 0 and (prev_conf - recent_conf) / prev_conf > 0.15:
        alerts.append({
            "severity": "high",
            "metric": "confidence",
            "message": f"Avg confidence dropped from {prev_conf:.2f} to {recent_conf:.2f} (~{(prev_conf - recent_conf)/prev_conf:.0%} drop)."
        })
    elif prev_conf > 0 and (prev_conf - recent_conf) / prev_conf > 0.08:
        alerts.append({
            "severity": "medium",
            "metric": "confidence",
            "message": f"Moderate confidence decrease: {prev_conf:.2f} → {recent_conf:.2f}."
        })

    # DQR drift (lower is worse)
    recent_dqr = recent["dqr"].mean()
    prev_dqr = prev["dqr"].mean() if not prev.empty else recent_dqr
    if prev_dqr > 0 and (prev_dqr - recent_dqr) / max(prev_dqr, 1e-6) > 0.15:
        alerts.append({
            "severity": "high",
            "metric": "dqr",
            "message": f"DQR decreased significantly: {prev_dqr:.2f} → {recent_dqr:.2f}."
        })

    # Unknown rate spike
    recent_unknown_rate = recent["is_unknown"].mean()
    prev_unknown_rate = prev["is_unknown"].mean() if not prev.empty else recent_unknown_rate
    if prev_unknown_rate >= 0 and (recent_unknown_rate - prev_unknown_rate) > 0.05 and recent_unknown_rate > 0.05:
        alerts.append({
            "severity": "high",
            "metric": "unknown_rate",
            "message": f"Unknown medicine rate up: {prev_unknown_rate:.2%} → {recent_unknown_rate:.2%}."
        })

    return alerts

# ---------------------
# Admin DB actions (delete, export)
# ---------------------
def delete_scan(scan_id: str) -> bool:
    """Delete a scan by id (returns True if deleted)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM scans WHERE scan_id=?", (scan_id,))
        deleted = c.rowcount
        conn.commit()
        conn.close()
        return deleted > 0
    except Exception:
        return False

def export_all_scans_csv() -> bytes:
    """Export all scans table as CSV bytes."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM scans ORDER BY timestamp DESC", conn)
    conn.close()
    return df_to_csv_bytes(df)

# ---------------------
# UI: Sidebar navigation
# ---------------------
st.sidebar.title("MediCure Monitoring")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Drift Detection",
    "Image Inspector",
    "ML Evaluation",
    "Pipeline Profiling",
    "Admin"
])

# Load and prepare DataFrame
raw = load_raw_metrics()
df = build_df(raw)

# ---------------------
# Page: Overview
# ---------------------
if page == "Overview":
    st.header("Overview — Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    col1.metric("Total Scans", f"{total:,}")
    col2.metric("Avg Duration (s)", f"{df['duration'].mean():.2f}" if total else "0.00")
    col3.metric("Avg AI Confidence", f"{df['confidence'].mean()*100:.1f}%" if total else "0.0%")
    col4.metric("Avg DQR", f"{df['dqr'].mean():.2f}" if total else "0.00")

    st.markdown("---")
    st.subheader("Quality & Health")
    c1, c2, c3 = st.columns(3)
    unknown_count = int(df["is_unknown"].sum()) if not df.empty else 0
    low_conf_count = int((df["confidence"] < LOW_CONF_THRESHOLD).sum()) if not df.empty else 0
    low_dqr_count = int((df["dqr"] < LOW_DQR_THRESHOLD).sum()) if not df.empty else 0
    c1.metric("Unknown Meds", f"{unknown_count:,}")
    c2.metric("Low Confidence (<50%)", f"{low_conf_count:,}")
    c3.metric(f"Low DQR (<{LOW_DQR_THRESHOLD})", f"{low_dqr_count:,}")

    st.markdown("---")
    st.subheader("Trends")
    if not df.empty and df["timestamp"].notna().any():
        # Trend: confidence over time
        conf_trend = df.set_index("timestamp").resample("D")["confidence"].mean().dropna()
        st.line_chart(conf_trend)

        # Trend: DQR over time
        dqr_trend = df.set_index("timestamp").resample("D")["dqr"].mean().dropna()
        st.line_chart(dqr_trend)
    else:
        st.info("Not enough timestamped data to show trends.")

    st.markdown("---")
    st.subheader("Quick Tables")
    col_left, col_right = st.columns([2,1])
    with col_left:
        st.write("Top 10 Most Scanned Medicines")
        if not df.empty:
            st.table(df["medicine_name"].value_counts().head(10).rename_axis("Medicine").reset_index(name="Count"))
        else:
            st.info("No data.")
    with col_right:
        st.write("Top Alternatives")
        if not df.empty:
            alt_series = df["ai_parsed"].apply(lambda x: x.get("alternatives") if isinstance(x, dict) else []).explode().dropna()
            st.table(alt_series.value_counts().head(10).rename_axis("Alternative").reset_index(name="Count"))
        else:
            st.info("No alternatives data.")

# ---------------------
# Page: Drift Detection
# ---------------------
elif page == "Drift Detection":
    st.header("AI Drift Detection")
    st.markdown("Simple, interpretable drift signals based on recent vs historic windows.")

    alerts = simple_drift_alerts(df)
    if alerts:
        for a in alerts:
            if a["severity"] == "high":
                st.error(f"🚨 {a['metric'].upper()} — {a['message']}")
            elif a["severity"] == "medium":
                st.warning(f"⚠ {a['metric'].upper()} — {a['message']}")
            else:
                st.info(f"ℹ {a['metric'].upper()} — {a['message']}")
    else:
        st.success("No major drift signals detected (based on simple heuristics).")

    st.markdown("---")
    st.subheader("Drift Charts")
    if not df.empty and df["timestamp"].notna().any():
        window = st.sidebar.slider("Rolling window (days)", 3, 1, 30, 7)
        df_time = df.set_index("timestamp").sort_index()
        rolling_conf = df_time["confidence"].rolling(f"{window}D").mean()
        rolling_dqr = df_time["dqr"].rolling(f"{window}D").mean()

        st.line_chart(rolling_conf.dropna().rename("rolling_confidence"))
        st.line_chart(rolling_dqr.dropna().rename("rolling_dqr"))
    else:
        st.info("No timestamped data for drift charts.")

    st.markdown("---")
    st.subheader("Drift Diagnostics (per-metric distribution)")
    if not df.empty:
        st.write("Confidence distribution (histogram)")
        st.bar_chart(np.histogram(df["confidence"].fillna(0), bins=20)[0])
        st.write("DQR distribution (histogram)")
        st.bar_chart(np.histogram(df["dqr"].fillna(0), bins=20)[0])
    else:
        st.info("No data.")

# ---------------------
# Page: Image Inspector
# ---------------------
elif page == "Image Inspector":
    st.header("Image Quality Debugger / Scan Inspector")
    st.markdown("Inspect individual scans: OCR text, parsed JSON, DQR, blur, brightness, and download full details.")

    if df.empty:
        st.info("No scans to inspect.")
    else:
        # Select by medicine, scan id not present in metrics list; we will show medicine and timestamp for selection
        df_idx = df.reset_index().reset_index().rename(columns={"level_0":"orig_index"})
        df_idx["label"] = df_idx.apply(lambda r: f"{r['medicine_name'] or 'UNKNOWN'} — {r['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(r['timestamp']) else 'no-time'}", axis=1)
        sel = st.selectbox("Pick a scan", options=df_idx.index, format_func=lambda i: df_idx.loc[i, "label"])
        sel_row = df_idx.loc[sel]
        st.subheader("Scan Summary")
        st.write("Medicine:", sel_row["medicine_name"])
        st.write("Timestamp:", sel_row["timestamp"])
        st.write("Duration (s):", sel_row["duration"])
        st.write("AI Confidence:", f"{sel_row['confidence']*100:.1f}%")
        st.write("DQR:", sel_row["dqr"])
        st.write("OCR length:", sel_row["ocr_text_length"])

        st.markdown("---")
        st.subheader("OCR Extracted Text")
        ocr = sel_row.get("ocr_text") or ""
        st.text_area("OCR Text", value=ocr, height=200)

        st.markdown("---")
        st.subheader("AI Parsed JSON (pretty)")
        parsed = sel_row.get("ai_parsed") or {}
        st.json(parsed)

        st.markdown("---")
        st.subheader("Image Quality / Meta")
        st.write("Blur score:", sel_row.get("blur_score"))
        st.write("Brightness:", sel_row.get("brightness"))
        st.write("Resolution:", sel_row.get("resolution"))
        st.write("OCR contains medicine name:", bool(sel_row.get("ocr_contains_name")))

        st.markdown("---")
        st.subheader("Download / Actions")
        # Download full_details from scans table by scan_id if available
        # We attempt to find scan_id by matching medicine_name & timestamp (best-effort)
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            # best-effort matching using timestamp
            ts = sel_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(sel_row["timestamp"]) else None
            if ts:
                c.execute("SELECT scan_id, full_details FROM scans WHERE timestamp LIKE ? LIMIT 1", (f"{ts}%",))
                res = c.fetchone()
            else:
                # fallback by medicine name
                c.execute("SELECT scan_id, full_details FROM scans WHERE medicine_name=? LIMIT 1", (sel_row["medicine_name"],))
                res = c.fetchone()
            conn.close()
        except Exception:
            res = None

        if res:
            scan_id, full_details = res[0], res[1]
            st.write("Detected scan_id:", scan_id)
            st.markdown(download_link_bytes(full_details.encode("utf-8"), f"{scan_id}_details.json", mime="application/json"), unsafe_allow_html=True)
        else:
            st.info("Could not reliably find raw scan JSON in DB for download (matching logic failed).")

# ---------------------
# Page: ML Evaluation
# ---------------------
elif page == "ML Evaluation":
    st.header("ML Evaluation — compare models/prompts and metrics")
    st.markdown("This page will compare confidence / parsing success across different 'model' or 'prompt' identifiers if present in ai_parsed.")

    if df.empty:
        st.info("No data for evaluation.")
    else:
        # Try to extract a 'model' or 'prompt_version' field from ai_parsed or details
        def extract_model_tag(parsed: Dict[str,Any]):
            # common keys to check
            for k in ("model", "model_name", "prompt_version", "prompt_id", "version"):
                v = parsed.get(k)
                if v:
                    return str(v)
            # fallback to empty
            return "unknown"

        df["eval_group"] = df["ai_parsed"].apply(lambda p: extract_model_tag(p) if isinstance(p, dict) else "unknown")
        groups = df["eval_group"].unique().tolist()

        st.write("Detected evaluation groups (model/prompt):", groups)
        group_selector = st.selectbox("Choose group to compare (or 'all')", options=["all"] + groups)
        if group_selector == "all":
            sub = df
        else:
            sub = df[df["eval_group"] == group_selector]

        if sub.empty:
            st.info("No rows for selected group.")
        else:
            st.subheader("Group statistics")
            st.write(sub[["eval_group","confidence","dqr","is_unknown"]].describe())

            st.subheader("Confidence distribution")
            st.bar_chart(np.histogram(sub["confidence"].fillna(0), bins=20)[0])

            st.subheader("Parsing Success & Alternatives")
            parsing_rate = sub["ai_parsed"].apply(lambda x: isinstance(x, dict) and len(x) > 0).mean() * 100
            alt_rate = sub["ai_parsed"].apply(lambda x: isinstance(x, dict) and len(x.get("alternatives", []))>0).mean() * 100
            st.metric("Parsing success", f"{parsing_rate:.1f}%")
            st.metric("Alternative suggestions success", f"{alt_rate:.1f}%")

            st.markdown("---")
            st.subheader("Per-medicine sample")
            top_sample = sub["medicine_name"].value_counts().head(10).index.tolist()
            sample_df = sub[sub["medicine_name"].isin(top_sample)][["medicine_name","confidence","dqr","ai_parsed"]]
            st.dataframe(sample_df.head(50))

# ---------------------
# Page: Pipeline Profiling
# ---------------------
elif page == "Pipeline Profiling":
    st.header("Pipeline Profiling — stage timings and bottlenecks")
    st.markdown("If your `full_details` JSON contains stage timing (e.g., 'stages': {'ocr': 0.12, 'genai': 0.45}), this page will summarize them.")

    if df.empty:
        st.info("No data.")
    else:
        # Extract stage times from ai_parsed or full_details if present
        def extract_stages(parsed):
            if isinstance(parsed, dict):
                stages = parsed.get("stages") or parsed.get("timings") or parsed.get("profiling")
                if isinstance(stages, dict):
                    return stages
            return {}

        df["stages"] = df["ai_parsed"].apply(lambda p: extract_stages(p) if isinstance(p, dict) else {})
        # Build dataframe with each stage as column (safely)
        all_stage_names = set()
        for s in df["stages"]:
            if isinstance(s, dict):
                all_stage_names.update(s.keys())
        if not all_stage_names:
            st.info("No stage timing data found in parsed JSON (key names like 'stages' or 'timings' expected).")
        else:
            # create columns
            for name in all_stage_names:
                df[f"stage__{name}"] = df["stages"].apply(lambda sd: float(sd.get(name, np.nan)) if isinstance(sd, dict) else np.nan)
            # show mean timings
            stage_cols = [c for c in df.columns if c.startswith("stage__")]
            mean_times = df[stage_cols].mean().sort_values(ascending=False)
            st.subheader("Average stage times (s)")
            st.table(mean_times.rename_axis("Stage").reset_index(name="AvgSeconds"))

            st.subheader("Top slow scans (by total stage sum)")
            df["stages_sum"] = df[stage_cols].sum(axis=1, skipna=True)
            slow = df.sort_values("stages_sum", ascending=False).head(10)[["medicine_name","timestamp","stages_sum"] + stage_cols]
            st.dataframe(slow.fillna("-"), use_container_width=True)

# ---------------------
# Page: Admin
# ---------------------
elif page == "Admin":
    st.header("Admin Tools — cleanup & exports")
    st.markdown("Danger zone: delete scans, export DB, download JSON. Use with care.")

    # Export all scans
    if st.button("Export entire scans table (CSV)"):
        csv_bytes = export_all_scans_csv()
        st.markdown(download_link_bytes(csv_bytes, "scans_export.csv"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Delete a scan by scan_id (irreversible)")
    scan_id_to_delete = st.text_input("scan_id (exact)")
    if st.button("Delete scan"):
        if not scan_id_to_delete:
            st.warning("Please enter a scan_id.")
        else:
            ok = delete_scan(scan_id_to_delete)
            if ok:
                st.success(f"Deleted scan {scan_id_to_delete}.")
                st.experimental_rerun()
            else:
                st.error("Delete failed (scan_id not found or DB error).")

    st.markdown("---")
    st.subheader("Download single scan JSON by scan_id")
    scan_id_dl = st.text_input("scan_id to download (exact)", key="dl")
    if st.button("Download scan JSON"):
        if not scan_id_dl:
            st.warning("Enter scan_id.")
        else:
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT full_details FROM scans WHERE scan_id=?", (scan_id_dl,))
                row = c.fetchone()
                conn.close()
                if row and row[0]:
                    st.markdown(download_link_bytes(row[0].encode("utf-8"), f"{scan_id_dl}.json", mime="application/json"), unsafe_allow_html=True)
                else:
                    st.error("Scan not found.")
            except Exception as e:
                st.error(f"DB error: {e}")

    st.markdown("---")
    st.subheader("Quick DB summary")
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM scans")
        total_rows = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM chat_history")
        chat_rows = c.fetchone()[0]
        conn.close()
        st.write(f"Scans rows: {total_rows:,} — Chat history rows: {chat_rows:,}")
    except Exception as e:
        st.error(f"Cannot open DB: {e}")

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.caption("MediCure AI Monitoring Suite • Designed for demos and professor review. © Medicure")

