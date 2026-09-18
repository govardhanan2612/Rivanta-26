"""Turning flagged points into incidents, and explaining them.

A flagged point on its own is not useful to anyone on call. What matters is
the event: when it started, how long it ran, which metrics moved, whether
they moved together, and what shape the movement had.

The explanation is assembled from measurements taken on the incident window -
contribution shares, a fitted trend, a correlation between residuals - rather
than from a lookup on which metric names happen to be involved. The shape and
the co-movement are what drive the hypothesis; the metric names only colour
the wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from detection import METRIC_LABELS, METRICS, severity_label

# Flagged points this far apart still belong to the same event. Detectors
# routinely drop a point mid-incident; splitting on that would produce two
# incidents where an operator sees one.
GAP_TOLERANCE = 3

# Context drawn either side of the incident when plotting and when measuring
# whether the metric came back to baseline afterwards.
CONTEXT = 25

# Persistence. A threshold crossing on a single sample is what noise looks
# like at any sensible sigma - over four metrics and a few hundred samples,
# some points cross by arithmetic alone. Requiring the departure to survive
# consecutive samples is the standard alerting remedy, and it is what keeps
# a clean run genuinely empty rather than merely quieter.
MIN_POINTS = 3

# Except this far out, where one sample is already past argument.
CONFIRM_SIGMA = 8.0

# A metric has to own a real share of the deviation *and* be meaningfully
# displaced on its own before it is named as a driver.
MIN_DRIVER_SHARE = 0.20
MIN_DRIVER_SIGMA = 2.0

# Above this, two residual series are treated as moving together.
CO_MOVEMENT_R = 0.70

# Share of the total departure delivered by its single largest step. Above
# this the fault arrived; below it, the fault accumulated.
ARRIVAL_JUMP = 0.5


@dataclass
class Incident:
    id: str
    start: pd.Timestamp
    end: pd.Timestamp
    start_index: int
    end_index: int
    onset_index: int
    point_count: int
    peak_sigma: float
    severity: str
    drivers: list[str]
    contributions: dict[str, float]
    directions: dict[str, int]
    ratios: dict[str, float]
    peaks: dict[str, float]
    shape: str
    correlation: float | None
    recovered: bool
    headline: str
    evidence: list[str] = field(default_factory=list)
    hypothesis: str = ""
    signature: str = ""
    occurrence: int = 1

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60) + 1

    @property
    def lead_minutes(self) -> int:
        """Minutes between the departure starting and the threshold tripping."""
        return self.start_index - self.onset_index

    @property
    def window_label(self) -> str:
        return f"{self.start:%H:%M}–{self.end:%H:%M}"


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def build_incidents(frame: pd.DataFrame, gap: int = GAP_TOLERANCE) -> list[Incident]:
    """Group contiguous runs of flagged points into explained incidents."""
    flagged = np.flatnonzero(frame["flagged"].to_numpy())
    if flagged.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = previous = int(flagged[0])
    for index in flagged[1:]:
        index = int(index)
        if index - previous <= gap:
            previous = index
            continue
        runs.append((start, previous))
        start = previous = index
    runs.append((start, previous))

    incidents = [
        _describe(frame, number, start, end)
        for number, (start, end) in enumerate(_persistent(frame, runs), start=1)
    ]
    _mark_recurrences(incidents)
    return incidents


def _persistent(frame: pd.DataFrame, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop runs too short to be anything but noise, unless they are extreme."""
    return [
        (start, end)
        for start, end in runs
        if (end - start + 1) >= MIN_POINTS
        or float(frame["deviation"].iloc[start : end + 1].max()) >= CONFIRM_SIGMA
    ]


def _describe(frame: pd.DataFrame, number: int, start: int, end: int) -> Incident:
    core = frame.iloc[start : end + 1]
    lead_in = max(0, start - CONTEXT)
    tail_out = min(len(frame) - 1, end + CONTEXT)
    trailing = frame.iloc[end + 1 : tail_out + 1]

    contributions = _contributions(core)
    drivers = _drivers(core, contributions)
    directions = {
        metric: int(np.sign(core[f"z_{metric}"].mean()) or 1) for metric in METRICS
    }
    ratios = _ratios(core)

    lead_metric = drivers[0] if drivers else str(core["worst_metric"].iloc[0])

    # Shape and co-movement are measured from onset, not from the threshold
    # crossing, so a gradual fault is judged on its whole climb. Both share a
    # single reference taken just before onset: mixing anchors makes two
    # measurements disagree about where the fault began.
    onset = _onset(frame, lead_metric, start)
    anchor = max(0, onset - 1)
    shape = _classify_shape(
        _reference_residual(frame, lead_metric, onset, end, anchor=anchor),
        core_length=end - start + 1,
    )
    correlation = (
        _co_movement(frame, drivers[0], drivers[1], onset, end, anchor)
        if len(drivers) >= 2
        else None
    )
    recovered = _has_recovered(trailing, lead_metric)

    peak_sigma = float(core["deviation"].max())
    # Measured from onset rather than from the threshold crossing: how long the
    # system was actually degraded is the question severity is asking, and the
    # crossing is only when we noticed.
    degraded = (
        int(
            (core["timestamp"].iloc[-1] - frame["timestamp"].iloc[onset]).total_seconds()
            // 60
        )
        + 1
    )
    peaks = {metric: float(np.abs(core[f"z_{metric}"]).max()) for metric in METRICS}

    incident = Incident(
        id=f"INC-{number:02d}",
        start=core["timestamp"].iloc[0],
        end=core["timestamp"].iloc[-1],
        start_index=start,
        end_index=end,
        onset_index=onset,
        point_count=len(core),
        peak_sigma=peak_sigma,
        severity=severity_label(peak_sigma, degraded),
        drivers=drivers or [lead_metric],
        contributions=contributions,
        directions=directions,
        ratios=ratios,
        peaks=peaks,
        shape=shape,
        correlation=correlation,
        recovered=recovered,
        headline="",
    )
    incident.headline = _headline(incident)
    incident.evidence = _evidence(incident)
    incident.hypothesis = _hypothesis(incident)
    incident.signature = _signature(incident)
    return incident


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def _contributions(core: pd.DataFrame) -> dict[str, float]:
    """Each metric's share of the total deviation across the incident."""
    strength = {
        metric: float(np.abs(core[f"z_{metric}"]).mean()) for metric in METRICS
    }
    total = sum(strength.values()) or 1.0
    return {metric: value / total for metric, value in strength.items()}


def _drivers(core: pd.DataFrame, contributions: dict[str, float]) -> list[str]:
    """Metrics that both own a share of the deviation and are displaced.

    The second test is what stops the least-quiet metric in an otherwise calm
    incident from being reported as a cause.
    """
    candidates = [
        metric
        for metric in METRICS
        if contributions[metric] >= MIN_DRIVER_SHARE
        and float(np.abs(core[f"z_{metric}"]).mean()) >= MIN_DRIVER_SIGMA
    ]
    return sorted(candidates, key=lambda metric: -contributions[metric])


def _ratios(core: pd.DataFrame) -> dict[str, float]:
    """Incident mean over trailing-baseline mean, per metric."""
    ratios = {}
    for metric in METRICS:
        baseline = float(core[f"center_{metric}"].mean())
        ratios[metric] = float(core[metric].mean()) / baseline if baseline else float("inf")
    return ratios


def _onset(frame: pd.DataFrame, metric: str, start: int, floor: float = 1.5) -> int:
    """Walk back to where the departure actually began.

    The threshold crossing is when the alarm fired, not when the fault
    started; for anything gradual those are different moments. The earlier one
    is the more useful timestamp for an operator, and it is also the one that
    makes the shape legible - measuring a ramp from the point it tripped the
    alarm cuts off the climb that identifies it.

    This walk reads the live trailing z-score, not a frozen reference. The
    question it asks is 'was this point already abnormal at the time', and a
    reference frozen further back would count ordinary cyclical drift across
    the lead-in as departure, dragging onset - and with it the measured
    duration - minutes earlier than the truth.
    """
    limit = max(0, start - CONTEXT)
    scores = frame[f"z_{metric}"]
    index = start
    while index > limit and abs(float(scores.iloc[index - 1])) >= floor:
        index -= 1
    return index


def _reference_residual(
    frame: pd.DataFrame, metric: str, start: int, end: int, anchor: int | None = None
) -> np.ndarray:
    """Departure across the incident, against the baseline frozen at onset.

    Measuring shape against the *live* trailing baseline is self-defeating:
    during a slow climb the baseline climbs along with the metric, so the
    residual flattens and a genuine ramp reads as a plateau. Anchoring the
    reference to the moment before the incident is what keeps the shape - and
    the correlation between two metrics' shapes - visible.
    """
    anchor = max(0, start - 1) if anchor is None else max(0, anchor)
    center = float(frame[f"center_{metric}"].iloc[anchor])
    spread = float(frame[f"spread_{metric}"].iloc[anchor])
    values = frame[metric].iloc[start : end + 1].to_numpy()
    return (values - center) / max(spread, 1e-9)


def _classify_shape(residual: np.ndarray, core_length: int) -> str:
    """Name the shape of a run from its own geometry, not from its metric.

    The question that separates the cases is whether the departure *arrived*
    or *accumulated*, so that is what gets measured: the largest single-sample
    step, against the size of the departure as a whole. A spike puts nearly
    all of its displacement into one step; a leak spreads it across many.
    Comparing window halves cannot do this - a run that opens at baseline and
    closes high looks like growth whether it crept there or jumped.
    """
    peak = float(np.abs(residual).max())
    if peak == 0:
        return "flat"

    # The leading zero matters: the reference is anchored before the incident,
    # so the first sample's own displacement is itself an arrival step.
    steps = np.diff(np.concatenate(([0.0], residual)))
    arrival = float(np.abs(steps).max()) / peak

    if arrival < ARRIVAL_JUMP:
        return "ramp"
    return "spike" if core_length < 5 else "sustained"


def _co_movement(
    frame: pd.DataFrame, first: str, second: str, start: int, end: int, anchor: int
) -> float | None:
    """Correlation between two metrics' departures across the incident."""
    if end - start + 1 < 4:
        return None
    left = _reference_residual(frame, first, start, end, anchor=anchor)
    right = _reference_residual(frame, second, start, end, anchor=anchor)
    if left.std() < 1e-9 or right.std() < 1e-9:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _has_recovered(trailing: pd.DataFrame, metric: str) -> bool:
    """Did the lead metric return to inside its band after the incident?"""
    if trailing.empty:
        return False
    return bool(np.abs(trailing[f"z_{metric}"]).head(10).max() < 2.0)


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------
SHAPE_PHRASES = {
    "ramp": "climbed steadily",
    "sustained": "held off-baseline",
    "spike": "jumped",
    "flat": "moved",
}


def _direction_word(direction: int, shape: str) -> str:
    if direction > 0:
        return SHAPE_PHRASES[shape]
    return {"ramp": "fell away steadily", "sustained": "stayed depressed", "spike": "dropped", "flat": "moved"}[shape]


def _headline(incident: Incident) -> str:
    leads = incident.drivers[:2]
    names = [METRIC_LABELS[metric] for metric in leads]
    action = _direction_word(incident.directions[leads[0]], incident.shape)
    if len(leads) < 2:
        return f"{names[0]} {action}"
    if incident.directions[leads[0]] != incident.directions[leads[1]]:
        return f"{names[0]} and {names[1]} moved in opposite directions"
    return f"{names[0]} and {names[1]} {action} together"


def _evidence(incident: Incident) -> list[str]:
    lines = []
    for metric in incident.drivers[:3]:
        direction = "above" if incident.directions[metric] > 0 else "below"
        share = incident.contributions[metric] * 100
        # Sigma leads, because sigma is what the detector actually decided on.
        # The ratio is easier to picture but flatters metrics with a wide
        # baseline, so it follows rather than leads, and only when it says
        # something a reader could not already see.
        ratio = incident.ratios[metric]
        as_ratio = f", {ratio:.2f}× normal" if abs(ratio - 1) >= 0.08 else ""
        lines.append(
            f"{METRIC_LABELS[metric]} peaked {incident.peaks[metric]:.1f}σ {direction} "
            f"its trailing baseline{as_ratio}, accounting for {share:.0f}% of the deviation."
        )

    lines.append(
        f"Peak departure {incident.peak_sigma:.1f}σ over {incident.duration_minutes} min "
        f"({incident.point_count} points flagged)."
    )

    if incident.lead_minutes > 0:
        onset_time = incident.start - pd.Timedelta(minutes=incident.lead_minutes)
        lines.append(
            f"The departure began around {onset_time:%H:%M}, {incident.lead_minutes} min "
            f"before it crossed the alerting threshold."
        )

    if incident.correlation is not None:
        strength = "moved together" if abs(incident.correlation) >= CO_MOVEMENT_R else "moved largely independently"
        lines.append(
            f"Residuals of the two lead metrics {strength} (r = {incident.correlation:.2f})."
        )

    shape_note = {
        "ramp": "The departure grew across the window rather than arriving at once.",
        "sustained": "The departure arrived quickly and then held at a level.",
        "spike": "The departure was brief and did not persist.",
        "flat": "",
    }[incident.shape]
    if shape_note:
        lines.append(shape_note)

    lines.append(
        "Metrics returned inside the band afterwards."
        if incident.recovered
        else "Metrics had not returned inside the band by the end of the context window."
    )
    return lines


def _hypothesis(incident: Incident) -> str:
    """A cause suggested by the measured geometry, offered as a hypothesis.

    The branch conditions are shape, co-movement and direction - all measured.
    Metric identity only refines the wording once a branch is chosen, so a
    different set of metrics showing the same geometry reaches the same class
    of conclusion.
    """
    rising = [m for m in incident.drivers if incident.directions[m] > 0]
    falling = [m for m in incident.drivers if incident.directions[m] < 0]
    together = incident.correlation is not None and abs(incident.correlation) >= CO_MOVEMENT_R

    if incident.shape == "ramp" and together and len(rising) >= 2:
        return (
            "Resource exhaustion. Two metrics climbing on a common gradient, rather "
            "than stepping up at once, is what accumulation looks like — a leak, an "
            "unbounded cache, or a queue filling faster than it drains. Check "
            "allocation and queue depth over the whole window, not just at the peak."
        )

    if incident.shape == "ramp" and len(rising) == 1:
        return (
            f"Gradual saturation of {METRIC_LABELS[rising[0]].lower()}. A single "
            "metric on a steady gradient usually means demand growing against a "
            "fixed ceiling. Worth extrapolating: the trend, not the current value, "
            "is what sets the time to failure."
        )

    if incident.shape == "spike" and len(rising) == 1 and incident.recovered:
        return (
            f"Transient load on {METRIC_LABELS[rising[0]].lower()}. It arrived and "
            "cleared without dragging other metrics with it, which points at a "
            "short-lived job or a burst of traffic rather than a structural fault. "
            "Only worth chasing if it recurs."
        )

    if falling and not rising:
        return (
            "Loss of throughput. The signal dropping rather than rising points "
            "upstream — a failed dependency, a health check pulling an instance "
            "out of rotation, or a network path going away. Confirm whether the "
            "work stopped arriving or stopped being served."
        )

    if len(incident.drivers) >= 2 and not together:
        return (
            "Multiple metrics are off baseline but not in step, so this is more "
            "likely two overlapping events than one cause. Worth splitting the "
            "window and triaging the metrics separately."
        )

    return (
        "No single pattern fits cleanly. The measurements above stand on their "
        "own; treat the cause as open until the window is correlated against "
        "deploys and upstream events."
    )


def _signature(incident: Incident) -> str:
    direction = "up" if incident.directions[incident.drivers[0]] > 0 else "down"
    return f"{incident.shape}:{direction}:{'+'.join(sorted(incident.drivers))}"


def _mark_recurrences(incidents: list[Incident]) -> None:
    """Number repeat occurrences of the same measured signature.

    Deliberately a deterministic grouping rather than a clustering step:
    with a handful of incidents in a window, k-means would be fitting noise,
    and an operator gets more from 'this is the third time' than from a
    cluster id.
    """
    seen: dict[str, int] = {}
    for incident in incidents:
        seen[incident.signature] = seen.get(incident.signature, 0) + 1
        incident.occurrence = seen[incident.signature]


def recurring_patterns(incidents: list[Incident]) -> list[dict]:
    """Signatures seen more than once, most frequent first."""
    groups: dict[str, list[Incident]] = {}
    for incident in incidents:
        groups.setdefault(incident.signature, []).append(incident)

    patterns = [
        {
            "signature": signature,
            "count": len(members),
            "headline": members[0].headline,
            "severity": max(members, key=lambda i: i.peak_sigma).severity,
            "windows": [member.window_label for member in members],
        }
        for signature, members in groups.items()
        if len(members) > 1
    ]
    return sorted(patterns, key=lambda pattern: -pattern["count"])


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def incidents_table(incidents: list[Incident]) -> pd.DataFrame:
    """Flat table of incidents, for the on-screen list and the CSV export."""
    return pd.DataFrame(
        [
            {
                "Incident": incident.id,
                "Start": incident.start.strftime("%Y-%m-%d %H:%M"),
                "Duration (min)": incident.duration_minutes,
                "Severity": incident.severity,
                "Peak (sigma)": round(incident.peak_sigma, 1),
                "Shape": incident.shape,
                "Drivers": ", ".join(METRIC_LABELS[m] for m in incident.drivers),
                "Summary": incident.headline,
                "Hypothesis": incident.hypothesis,
            }
            for incident in incidents
        ]
    )


def incident_report(incidents: list[Incident], source: str) -> str:
    """Plain-text report, the thing you would actually paste into a ticket."""
    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "ANOMALY INVESTIGATION REPORT",
        "=" * 60,
        f"Generated  {generated}",
        f"Source     {source}",
        f"Incidents  {len(incidents)}",
        "",
    ]
    for incident in incidents:
        lines.extend(
            [
                "-" * 60,
                f"{incident.id}  {incident.severity.upper()}  {incident.window_label}"
                f"  ({incident.duration_minutes} min)",
                "-" * 60,
                incident.headline,
                "",
                "Evidence:",
                *[f"  - {line}" for line in incident.evidence],
                "",
                "Hypothesis:",
                f"  {incident.hypothesis}",
                "",
            ]
        )
        if incident.occurrence > 1:
            lines.append(f"  Occurrence {incident.occurrence} of this signature.\n")
    return "\n".join(lines)
