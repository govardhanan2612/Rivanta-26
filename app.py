"""
Anomaly Detection & Root Cause Analysis - Hackathon Demo App
==============================================================
STEP 1: Synthetic data + Isolation Forest + line charts
STEP 2: Root cause (z-score) + severity scoring + alert text
STEP 3: Normal-range bands + summary metrics + clean table
STEP 4: Live feed simulation + clustering + incident report download

Run with:
    streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# The metric columns we always expect to work with.
# Keeping this as one constant makes it easy to loop over metrics
# everywhere else in the app instead of repeating column names.
FEATURE_COLUMNS = ["cpu_usage", "memory_usage", "response_time", "network_traffic"]


# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------------------
def generate_synthetic_data(n_points: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Build a fake "server monitoring" time series with a handful of
    realistic anomalies injected into it (spikes, dips, correlated
    multi-metric events like a memory-leak pattern).

    Returns a DataFrame with columns:
        timestamp, cpu_usage, memory_usage, response_time, network_traffic
    """
    rng = np.random.default_rng(seed)

    # One data point per minute, starting "now" minus n_points minutes.
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_points, freq="min")

    # --- Normal baseline behavior for each metric ---
    # A slow sine wave gives each metric a gentle daily-cycle feel,
    # and Gaussian noise makes it look like real sensor data.
    t = np.arange(n_points)

    cpu_usage = 40 + 10 * np.sin(t / 50) + rng.normal(0, 3, n_points)
    memory_usage = 55 + 8 * np.sin(t / 70 + 1) + rng.normal(0, 2.5, n_points)
    response_time = 200 + 20 * np.sin(t / 40 + 2) + rng.normal(0, 10, n_points)
    network_traffic = 500 + 100 * np.sin(t / 60) + rng.normal(0, 30, n_points)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "response_time": response_time,
            "network_traffic": network_traffic,
        }
    )

    # --- Inject a few realistic anomalies at fixed spots ---
    # Each entry is (index, description). We pick indices that are
    # comfortably inside the series so the "shape" is visible.

    # 1) CPU spike (e.g. runaway process)
    idx = int(n_points * 0.20)
    df.loc[idx : idx + 3, "cpu_usage"] += 45

    # 2) Correlated memory-leak pattern: memory + response_time climb together
    idx = int(n_points * 0.45)
    df.loc[idx : idx + 6, "memory_usage"] += np.linspace(10, 40, 7)
    df.loc[idx : idx + 6, "response_time"] += np.linspace(50, 180, 7)

    # 3) Network traffic drop (e.g. outage / connectivity loss)
    idx = int(n_points * 0.65)
    df.loc[idx : idx + 4, "network_traffic"] -= 350

    # 4) Sudden response_time spike (e.g. slow downstream dependency)
    idx = int(n_points * 0.85)
    df.loc[idx : idx + 2, "response_time"] += 400

    # Clip values so nothing goes unrealistically negative.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# 2. ANOMALY DETECTION (Isolation Forest)
# ---------------------------------------------------------------------------
def run_isolation_forest(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    """
    Fit an Isolation Forest on the feature columns and attach two new
    columns to a copy of df:
        anomaly        -> True/False
        anomaly_score   -> raw model score (lower = more abnormal)
    """
    df = df.copy()

    model = IsolationForest(
        contamination=contamination,  # expected fraction of anomalies ("sensitivity")
        random_state=42,
        n_estimators=200,
    )
    X = df[FEATURE_COLUMNS].values

    # predict(): -1 means anomaly, 1 means normal
    predictions = model.fit_predict(X)
    df["anomaly"] = predictions == -1

    # score_samples(): higher = more normal, lower = more anomalous.
    # We keep the raw score around for Step 2 (severity scoring).
    df["anomaly_score"] = model.score_samples(X)

    return df


# ---------------------------------------------------------------------------
# 2b. ROOT CAUSE ANALYSIS + SEVERITY SCORING
# ---------------------------------------------------------------------------
def compute_baseline_stats(df: pd.DataFrame) -> dict:
    """
    Compute the "normal" mean and standard deviation for each feature,
    using only the points the model did NOT flag as anomalies.
    This baseline is what we compare anomalies against.
    """
    normal_rows = df[~df["anomaly"]]
    stats = {}
    for feature in FEATURE_COLUMNS:
        mean = normal_rows[feature].mean()
        std = normal_rows[feature].std()
        # Guard against a zero std (would cause divide-by-zero below).
        if std == 0 or pd.isna(std):
            std = 1e-6
        stats[feature] = {"mean": mean, "std": std}
    return stats


def compute_severity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the Isolation Forest's raw anomaly_score into an easy-to-read
    0-100 "severity_score" (100 = most abnormal), and bucket it into
    Minor / Moderate / Critical labels.

    Isolation Forest scores: higher score = more normal, lower (more
    negative) score = more abnormal. We flip and min-max scale it.
    """
    df = df.copy()
    score_min = df["anomaly_score"].min()
    score_max = df["anomaly_score"].max()
    score_range = max(score_max - score_min, 1e-6)  # avoid divide-by-zero

    # Flip so higher = more abnormal, then scale to 0-100.
    df["severity_score"] = ((score_max - df["anomaly_score"]) / score_range * 100).round(1)

    def label_severity(score: float) -> str:
        if score >= 75:
            return "Critical"
        elif score >= 50:
            return "Moderate"
        else:
            return "Minor"

    df["severity_label"] = df["severity_score"].apply(label_severity)
    return df


def classify_pattern(top_features: list) -> str:
    """
    Very small heuristic that turns the set of most-deviating features
    into a human-friendly "possible X pattern" phrase for the alert text.
    """
    names = {f[0] for f in top_features}

    if {"memory_usage", "response_time"}.issubset(names):
        return "possible memory leak pattern"
    if "cpu_usage" in names and len(names) == 1:
        return "possible CPU overload pattern"
    if "network_traffic" in names and len(names) == 1:
        return "possible network outage pattern"
    if "response_time" in names and len(names) == 1:
        return "possible slow dependency pattern"
    return "possible multi-metric anomaly pattern"


def compute_root_cause(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    For every row, figure out which 1-2 features deviated the most from
    their normal baseline (using a z-score: how many std-devs away from
    the mean a value is). Attach a plain-English 'alert' message for
    each anomalous row.
    """
    df = df.copy()
    driving_features_col = []
    alert_col = []

    for _, row in df.iterrows():
        # Compute z-score and ratio-vs-normal for every feature on this row.
        deviations = []
        for feature in FEATURE_COLUMNS:
            mean = stats[feature]["mean"]
            std = stats[feature]["std"]
            z = (row[feature] - mean) / std
            ratio = row[feature] / mean if mean != 0 else float("inf")
            deviations.append((feature, z, ratio))

        # Sort by how extreme the z-score is (biggest absolute deviation first).
        deviations.sort(key=lambda x: abs(x[1]), reverse=True)
        top_two = deviations[:2]

        if not row["anomaly"]:
            driving_features_col.append("")
            alert_col.append("")
            continue

        # Build the "feature X.Xx above/below normal" phrases.
        phrases = []
        feature_names = []
        for feature, z, ratio in top_two:
            if abs(z) < 1:  # not meaningfully different from normal, skip it
                continue
            direction = "above" if z > 0 else "below"
            phrases.append(f"{feature} {ratio:.1f}x {direction} normal")
            feature_names.append(feature)

        if not phrases:  # fallback in case nothing stood out individually
            feature, z, ratio = top_two[0]
            direction = "above" if z > 0 else "below"
            phrases.append(f"{feature} {ratio:.1f}x {direction} normal")
            feature_names.append(feature)

        pattern = classify_pattern(top_two)
        timestamp_str = row["timestamp"].strftime("%H:%M")
        alert = (
            f"⚠️ {row['severity_label']}: " + ", ".join(phrases) +
            f" at {timestamp_str} — {pattern}."
        )

        driving_features_col.append(", ".join(feature_names))
        alert_col.append(alert)

    df["driving_features"] = driving_features_col
    df["alert"] = alert_col
    return df


# ---------------------------------------------------------------------------
# 3. CHARTING
# ---------------------------------------------------------------------------
def plot_feature(df: pd.DataFrame, feature: str, stats: dict) -> go.Figure:
    """Line chart for one metric, with anomalous points highlighted in red
    and a shaded "normal range" band (mean +/- 1 std dev) behind the line."""
    fig = go.Figure()

    # Shaded normal-range band so anomalies visually pop against it.
    mean = stats[feature]["mean"]
    std = stats[feature]["std"]
    fig.add_hrect(
        y0=mean - std,
        y1=mean + std,
        fillcolor="#4C78A8",
        opacity=0.15,
        line_width=0,
    )

    # Normal line (all points, drawn as a continuous line)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[feature],
            mode="lines",
            name=feature,
            line=dict(color="#4C78A8"),
        )
    )

    # Anomalous points, overlaid as red markers
    anomalies = df[df["anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies[feature],
            mode="markers",
            name="anomaly",
            marker=dict(color="red", size=9, symbol="circle-open", line=dict(width=2)),
        )
    )

    fig.update_layout(
        title=feature.replace("_", " ").title(),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 4a. CLUSTERING - group anomalies into patterns with KMeans
# ---------------------------------------------------------------------------
# Friendly names for the "X cluster" labels, keyed by the feature that
# deviates the most within that cluster.
CLUSTER_LABELS = {
    "cpu_usage": "CPU spike cluster",
    "memory_usage": "Memory growth cluster",
    "response_time": "Response time spike cluster",
    "network_traffic": "Network drop cluster",
}


def cluster_anomalies(df: pd.DataFrame, stats: dict, n_clusters: int = 3):
    """
    Group the anomalous points into clusters using KMeans on their
    z-score profile (how many std-devs each feature deviated by).
    Points that deviate in a similar way end up in the same cluster.

    Returns (anomalies_with_cluster_df, list_of_cluster_summaries) or
    (None, []) if there aren't enough anomalies to cluster.
    """
    anomalies = df[df["anomaly"]].copy()
    if len(anomalies) < 2:
        return None, []

    n_clusters = min(n_clusters, len(anomalies))

    # Build a matrix of z-scores: one row per anomaly, one column per feature.
    z_matrix = np.array(
        [
            [(row[f] - stats[f]["mean"]) / stats[f]["std"] for f in FEATURE_COLUMNS]
            for _, row in anomalies.iterrows()
        ]
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    anomalies["cluster"] = kmeans.fit_predict(z_matrix)

    # Name each cluster after whichever feature deviates the most on average.
    summaries = []
    for cluster_id in sorted(anomalies["cluster"].unique()):
        mask = anomalies["cluster"] == cluster_id
        avg_abs_z = np.abs(z_matrix[mask.values]).mean(axis=0)
        dominant_feature = FEATURE_COLUMNS[int(np.argmax(avg_abs_z))]
        summaries.append(
            {
                "cluster": cluster_id,
                "label": CLUSTER_LABELS[dominant_feature],
                "count": int(mask.sum()),
                "example_alert": anomalies[mask]["alert"].iloc[0],
            }
        )

    return anomalies, summaries


# ---------------------------------------------------------------------------
# 4b. LIVE FEED SIMULATION - genuinely re-runs detection as data streams in
# ---------------------------------------------------------------------------
def simulate_live_feed(raw_df: pd.DataFrame, contamination: float, delay: float = 0.03):
    """
    Simulate a live monitoring feed: data points are revealed a chunk at a
    time, and on EVERY chunk we re-fit a fresh Isolation Forest on only the
    data seen "so far" (raw_df.iloc[:i]) and recompute baseline stats,
    severity, and root cause from scratch on that window.

    This is genuine detection running incrementally, not a replay of
    labels computed once on the full dataset - early in the stream the
    model has less context, so its calls can differ from the final
    full-dataset pass, exactly like a real streaming detector would.
    """
    # Only the raw metric + timestamp columns go in - any anomaly/score
    # columns from a previous full-dataset run are dropped and recomputed.
    raw_df = raw_df[["timestamp", *FEATURE_COLUMNS]].reset_index(drop=True)

    metric_area = st.empty()
    chart_cols = st.columns(2)
    chart_areas = {
        feature: chart_cols[i % 2].empty() for i, feature in enumerate(FEATURE_COLUMNS)
    }
    alert_area = st.empty()

    # Isolation Forest needs a reasonable number of points to fit a
    # meaningful model, so we skip live detection until we have enough.
    min_points = max(20, int(1 / contamination))
    step = max(1, len(raw_df) // 80)  # re-fitting each frame is heavier than a replay, so fewer frames
    recent_alerts = []
    seen_alert_keys = set()

    for i in range(step, len(raw_df) + step, step):
        i = min(i, len(raw_df))
        partial_raw = raw_df.iloc[:i]

        if i < min_points:
            metric_area.metric("Anomalies detected so far", 0)
            for feature in FEATURE_COLUMNS:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=partial_raw["timestamp"], y=partial_raw[feature], mode="lines"))
                fig.update_layout(
                    title=f"{feature.replace('_', ' ').title()} (collecting baseline...)",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                )
                chart_areas[feature].plotly_chart(fig, width='stretch', key=f"live_{feature}_{i}")
            time.sleep(delay)
            continue

        # --- Real detection, re-run on exactly the data seen so far ---
        partial = run_isolation_forest(partial_raw, contamination)
        partial_stats = compute_baseline_stats(partial)
        partial = compute_severity_scores(partial)
        partial = compute_root_cause(partial, partial_stats)

        metric_area.metric("Anomalies detected so far", int(partial["anomaly"].sum()))

        for feature in FEATURE_COLUMNS:
            chart_areas[feature].plotly_chart(
                plot_feature(partial, feature, partial_stats),
                width='stretch',
                key=f"live_{feature}_{i}",
            )

        # Surface newly-seen alerts (dedupe by timestamp so a point that
        # keeps getting re-flagged each frame doesn't spam the feed).
        for _, row in partial[partial["anomaly"]].iterrows():
            key = row["timestamp"]
            if key not in seen_alert_keys:
                seen_alert_keys.add(key)
                recent_alerts.append(row["alert"])
        if recent_alerts:
            alert_area.info("\n\n".join(recent_alerts[-5:]))

        time.sleep(delay)

    st.success(f"✅ Live detection complete — streamed and re-scored {len(raw_df)} points.")


# ---------------------------------------------------------------------------
# 4. STREAMLIT APP
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Anomaly Detection & RCA", layout="wide")
    st.title("🔍 Anomaly Detection & Root Cause Analysis")
    st.caption("Isolation Forest over synthetic (or uploaded) system metrics")

    # --- Sidebar controls ---
    st.sidebar.header("Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (must contain: timestamp, cpu_usage, memory_usage, "
        "response_time, network_traffic)",
        type=["csv"],
    )

    if st.sidebar.button("🔄 Regenerate synthetic data"):
        st.session_state.pop("synthetic_df", None)

    # Load data: uploaded CSV takes priority, otherwise use (cached) synthetic data.
    required_columns = {"timestamp", *FEATURE_COLUMNS}
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        if not required_columns.issubset(df.columns):
            st.sidebar.error(
                "CSV is missing required columns: "
                + ", ".join(sorted(required_columns - set(df.columns)))
            )
            st.stop()
    else:
        if "synthetic_df" not in st.session_state:
            st.session_state["synthetic_df"] = generate_synthetic_data()
        df = st.session_state["synthetic_df"]

    st.sidebar.header("Detection Sensitivity")
    contamination = st.sidebar.slider(
        "Contamination (expected anomaly fraction)",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        help="Higher = more points flagged as anomalies (more sensitive).",
    )

    st.sidebar.header("Live Feed")
    playback_delay = st.sidebar.slider(
        "Stream speed (delay per frame, seconds)",
        min_value=0.0,
        max_value=0.1,
        value=0.02,
        step=0.01,
        help="The model is genuinely re-fit on each new chunk of data as it streams in.",
    )
    run_live_feed = st.sidebar.button("▶️ Start Live Detection")

    # --- Run detection ---
    raw_df = df.copy()  # keep the untouched raw data for the live-feed re-fit
    df = run_isolation_forest(df, contamination)

    # --- Root cause + severity (Step 2) ---
    baseline_stats = compute_baseline_stats(df)
    df = compute_severity_scores(df)
    df = compute_root_cause(df, baseline_stats)

    # --- Summary metrics (Step 3) ---
    total_points = len(df)
    total_anomalies = int(df["anomaly"].sum())
    pct_flagged = (total_anomalies / total_points * 100) if total_points else 0
    severity_counts = df[df["anomaly"]]["severity_label"].value_counts()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total anomalies", total_anomalies)
    m2.metric("% of data flagged", f"{pct_flagged:.1f}%")
    m3.metric("🟥 Critical", int(severity_counts.get("Critical", 0)))
    m4.metric("🟧 Moderate", int(severity_counts.get("Moderate", 0)))
    m5.metric("🟨 Minor", int(severity_counts.get("Minor", 0)))

    # --- Charts (each with a shaded "normal range" band) ---
    st.subheader("Metrics over time")
    if run_live_feed:
        simulate_live_feed(raw_df, contamination, delay=playback_delay)
    else:
        cols = st.columns(2)
        for i, feature in enumerate(FEATURE_COLUMNS):
            with cols[i % 2]:
                st.plotly_chart(
                    plot_feature(df, feature, baseline_stats), width='stretch'
                )

    # --- Clean anomaly table ---
    st.subheader("Detected anomalies")
    anomalies_df = df[df["anomaly"]][
        ["timestamp", "severity_label", "driving_features", "alert"]
    ].rename(
        columns={
            "timestamp": "Timestamp",
            "severity_label": "Severity",
            "driving_features": "Driving Feature(s)",
            "alert": "Explanation",
        }
    )
    st.dataframe(anomalies_df, width='stretch', height=300)

    # --- Anomaly clusters (Step 4) ---
    st.subheader("🧩 Anomaly Clusters")
    _, cluster_summaries = cluster_anomalies(df, baseline_stats)
    if cluster_summaries:
        cluster_cols = st.columns(len(cluster_summaries))
        for col, summary in zip(cluster_cols, cluster_summaries):
            with col:
                st.metric(summary["label"], f"{summary['count']} points")
                st.caption(summary["example_alert"])
    else:
        st.caption("Not enough anomalies yet to form clusters.")

    # --- Download incident report (Step 4) ---
    st.subheader("📄 Incident Report")
    report_csv = anomalies_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Incident Report (CSV)",
        data=report_csv,
        file_name="incident_report.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
