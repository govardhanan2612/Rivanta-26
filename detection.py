"""Metric simulation and the detection pipeline.

Detection runs two signals over the same data and combines them:

  1. A robust z-score against each metric's own *trailing* baseline. This is
     the primary signal. It is expressed in sigma, which is an absolute unit,
     so a score means the same thing regardless of what else is in the window.
  2. An Isolation Forest fitted on the z-scores. This is the corroborating
     signal, and its job is the points that look unremarkable metric by metric
     but sit in an odd place once you consider the metrics jointly.

Neither signal forces a fixed number of detections, so clean data produces an
empty result rather than a quota of false positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

METRICS = ("cpu_usage", "memory_usage", "response_time", "network_traffic")

METRIC_LABELS = {
    "cpu_usage": "CPU usage",
    "memory_usage": "Memory usage",
    "response_time": "Response time",
    "network_traffic": "Network traffic",
}

METRIC_UNITS = {
    "cpu_usage": "%",
    "memory_usage": "%",
    "response_time": "ms",
    "network_traffic": "KB/s",
}

# Trailing window used as the baseline, in samples (= minutes here). Long
# enough to ride out the ordinary load cycle, short enough that a genuine
# drift climbs out of it within a few minutes instead of being absorbed.
BASELINE_WINDOW = 45

# Converts a median-absolute-deviation into a standard deviation for
# normally distributed data. Using MAD rather than std keeps the baseline
# from being inflated by the very anomalies we are trying to find.
MAD_TO_SIGMA = 1.4826

# Severity cuts, in sigma. Fixed rather than relative to the current window,
# which is what makes severity stable as new data arrives.
CRITICAL_SIGMA = 10.0
MODERATE_SIGMA = 6.0

# A departure that holds for this long is escalated a level. Peak magnitude
# on its own undersells a sustained fault: a 7-sigma leak running eleven
# minutes is a worse morning than a 14-sigma blip that clears in four.
SUSTAINED_MINUTES = 10

# How far below zero the forest's decision function has to sit before its
# opinion counts as corroboration, and how far below the primary threshold
# that corroboration is allowed to reach.
FOREST_MARGIN = -0.04
CORROBORATED_FLOOR = 0.85


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_metrics(n_points: int = 500, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic server-metrics series with four faults injected.

    The faults deliberately differ in *shape*, not just magnitude, so the
    detector has to do more than find large numbers.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("min"), periods=n_points, freq="min"
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cpu_usage": 40 + 10 * np.sin(t / 50) + rng.normal(0, 3, n_points),
            "memory_usage": 55 + 8 * np.sin(t / 70 + 1) + rng.normal(0, 2.5, n_points),
            "response_time": 200 + 20 * np.sin(t / 40 + 2) + rng.normal(0, 10, n_points),
            "network_traffic": 500 + 100 * np.sin(t / 60) + rng.normal(0, 30, n_points),
        }
    )

    _inject_spike(frame, at=0.20, column="cpu_usage", amount=45, width=4)
    # A mild one. Real runs are not all sirens, and a triage queue where
    # everything is critical is not a triage queue.
    _inject_step(frame, at=0.32, column="response_time", amount=58, width=6)
    # The hard case: a slow climb across two metrics at once. No individual
    # early point is remarkable, which is exactly why a point-in-isolation
    # detector misses it and a trailing baseline does not.
    _inject_ramp(
        frame,
        at=0.44,
        columns=("memory_usage", "response_time"),
        amounts=(38.0, 170.0),
        width=14,
    )
    _inject_step(frame, at=0.66, column="network_traffic", amount=-340, width=6)
    _inject_spike(frame, at=0.85, column="response_time", amount=400, width=3)
    _inject_step(frame, at=0.93, column="cpu_usage", amount=30, width=4)

    frame[list(METRICS)] = frame[list(METRICS)].clip(lower=0)
    return frame


def _span(frame: pd.DataFrame, at: float, width: int) -> tuple[int, int]:
    start = int(len(frame) * at)
    return start, start + width - 1


def _inject_spike(frame, at, column, amount, width):
    start, end = _span(frame, at, width)
    frame.loc[start:end, column] += amount


def _inject_step(frame, at, column, amount, width):
    start, end = _span(frame, at, width)
    frame.loc[start:end, column] += amount


def _inject_ramp(frame, at, columns, amounts, width):
    start, end = _span(frame, at, width)
    for column, amount in zip(columns, amounts):
        frame.loc[start:end, column] += np.linspace(amount / width, amount, width)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def _trailing_baseline(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Rolling median and MAD-derived spread, computed from past points only.

    The shift(1) matters: without it a point contributes to the baseline it
    is being judged against, which quietly hides sustained faults.
    """
    past = series.shift(1)
    min_periods = max(5, window // 4)

    center = past.rolling(window, min_periods=min_periods).median()
    spread = (past - center).abs().rolling(window, min_periods=min_periods).median()
    spread = spread * MAD_TO_SIGMA

    # The first few points have no history to lean on, so they fall back to
    # whole-series statistics rather than being dropped.
    center = center.fillna(series.median())
    spread = spread.fillna((series - series.median()).abs().median() * MAD_TO_SIGMA)

    # A floor stops a briefly flat metric from producing an explosive z-score.
    floor = max(float(series.std(ddof=0)) * 0.05, 1e-6)
    return center, spread.clip(lower=floor)


def deviation_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Robust z-score per metric against its own trailing baseline."""
    scores = pd.DataFrame(index=frame.index)
    baseline: dict[str, dict[str, pd.Series]] = {}
    for metric in METRICS:
        center, spread = _trailing_baseline(frame[metric], BASELINE_WINDOW)
        scores[metric] = (frame[metric] - center) / spread
        baseline[metric] = {"center": center, "spread": spread}
    return scores, baseline


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect(frame: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Score and flag every point. Returns a copy with detection columns added."""
    frame = frame.reset_index(drop=True).copy()
    scores, baseline = deviation_scores(frame)

    # Worst single-metric departure, in sigma.
    peak = scores.abs().max(axis=1)
    worst = scores.abs().idxmax(axis=1)

    # The forest is fitted on z-scores rather than raw values, which makes it
    # scale-free across metrics and gives it the same temporal context the
    # primary signal has.
    forest = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
    )
    forest.fit(scores.to_numpy())

    # A clear margin below zero, not merely the sign of predict(). Borderline
    # points are the common case in healthy data, and treating them as
    # corroboration is how a multivariate signal turns into a noise amplifier.
    margin = forest.decision_function(scores.to_numpy())
    corroborated = margin <= FOREST_MARGIN

    # Flag on a clear single-metric departure, or on a slightly milder one that
    # the multivariate view also considers structurally odd. The corroborated
    # branch only reaches a little below the threshold on purpose: any lower
    # and it admits ordinary noise, which no amount of downstream grouping
    # can undo.
    flagged = (peak >= threshold) | (corroborated & (peak >= threshold * CORROBORATED_FLOOR))

    frame["deviation"] = peak
    frame["worst_metric"] = worst
    frame["flagged"] = flagged
    frame["corroborated"] = corroborated
    for metric in METRICS:
        frame[f"z_{metric}"] = scores[metric]
        frame[f"center_{metric}"] = baseline[metric]["center"]
        frame[f"spread_{metric}"] = baseline[metric]["spread"]
    return frame


def severity_label(sigma: float, duration_minutes: int = 1) -> str:
    """Severity from how far off baseline it went and how long it stayed."""
    ladder = ["Minor", "Moderate", "Critical"]
    if sigma >= CRITICAL_SIGMA:
        level = 2
    elif sigma >= MODERATE_SIGMA:
        level = 1
    else:
        level = 0

    if duration_minutes >= SUSTAINED_MINUTES:
        level = min(level + 1, 2)
    return ladder[level]


def validate_columns(frame: pd.DataFrame) -> list[str]:
    """Return the required columns this frame is missing."""
    required = {"timestamp", *METRICS}
    return sorted(required - set(frame.columns))
