"""Incident triage over system metrics.

Detection and explanation live in detection.py and incidents.py; this module
is layout and wiring only.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import theme
from detection import METRIC_LABELS, METRICS, detect, simulate_metrics, validate_columns
from incidents import (
    CONTEXT,
    build_incidents,
    incident_report,
    incidents_table,
    recurring_patterns,
)

st.set_page_config(
    page_title="Incident triage",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False, max_entries=64)
def analyse(frame: pd.DataFrame, threshold: float):
    """Detection plus incident assembly. Cached so slider moves are cheap."""
    scored = detect(frame, threshold)
    return scored, build_incidents(scored)


@st.cache_data(show_spinner=False)
def load_synthetic(seed: int) -> pd.DataFrame:
    return simulate_metrics(seed=seed)


def _prefers_dark() -> bool:
    """The viewer's own setting, where the running Streamlit exposes it."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def sidebar() -> tuple[pd.DataFrame, str, float]:
    st.sidebar.markdown("**Source**")
    upload = st.sidebar.file_uploader(
        "CSV with timestamp + " + ", ".join(METRICS),
        type=["csv"],
        label_visibility="collapsed",
    )

    if upload is not None:
        frame = pd.read_csv(upload, parse_dates=["timestamp"])
        missing = validate_columns(frame)
        if missing:
            st.sidebar.error("Missing columns: " + ", ".join(missing))
            st.stop()
        source = upload.name
    else:
        if st.sidebar.button("New sample run", width="stretch"):
            st.session_state.seed = st.session_state.get("seed", 42) + 1
            st.session_state.pop("selected", None)
        frame = load_synthetic(st.session_state.get("seed", 42))
        source = f"simulated run #{st.session_state.get('seed', 42)}"

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sensitivity**")
    threshold = st.sidebar.slider(
        "Flag at (σ from trailing baseline)",
        min_value=2.0, max_value=6.0, value=4.0, step=0.25,
        help=(
            "An absolute cut, not a quota. Nothing is flagged unless it actually "
            "departs from its own recent history by this much, so clean data "
            "returns no incidents. The default was chosen by sweeping this value "
            "against seeded clean and faulty runs: 4.0σ was the loosest setting "
            "that kept full recall of injected faults at under 0.2 false "
            "incidents per run."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Replay**")
    st.sidebar.caption(
        "Streams the run back and re-detects on each window, so you can watch "
        "a gradual fault cross the threshold."
    )
    if st.sidebar.button("Start replay", width="stretch"):
        st.session_state.live = True
        st.session_state.live_cursor = 60
    if st.session_state.get("live") and st.sidebar.button("Stop replay", width="stretch"):
        st.session_state.live = False

    st.sidebar.markdown("---")
    st.sidebar.toggle("Dark theme", key="dark")

    return frame, source, threshold


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def masthead(source: str, frame: pd.DataFrame) -> None:
    window = f"{frame['timestamp'].iloc[0]:%d %b %H:%M} – {frame['timestamp'].iloc[-1]:%H:%M}"
    st.markdown(
        f"""<div class="masthead">
          <h1>Incident triage</h1>
          <p>{len(frame)} points · {window} · {source}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def summary_strip(incidents, frame) -> None:
    critical = sum(1 for i in incidents if i.severity == "Critical")
    moderate = sum(1 for i in incidents if i.severity == "Moderate")
    minor = sum(1 for i in incidents if i.severity == "Minor")
    worst = max((i.peak_sigma for i in incidents), default=0.0)

    cells = [
        ("Incidents", f"{len(incidents)}", ""),
        ("Critical", f"{critical}", "critical" if critical else ""),
        ("Moderate", f"{moderate}", "moderate" if moderate else ""),
        ("Minor", f"{minor}", ""),
        ("Points flagged", f"{int(frame['flagged'].sum())}", ""),
        ("Worst departure", f"{worst:.1f}σ", ""),
    ]
    html = "".join(
        f'<div class="cell"><span class="value {tone}">{value}</span>'
        f'<span class="label">{label}</span></div>'
        for label, value, tone in cells
    )
    st.markdown(f'<div class="strip">{html}</div>', unsafe_allow_html=True)


def incident_rail(incidents) -> str:
    """Severity-ranked list. Returns the selected incident id."""
    ordered = sorted(incidents, key=lambda i: -i.peak_sigma)
    if st.session_state.get("selected") not in {i.id for i in ordered}:
        st.session_state.selected = ordered[0].id

    st.markdown('<p class="section">Incidents</p>', unsafe_allow_html=True)
    with st.container(height=560, border=False):
        for incident in ordered:
            label = (
                f"{incident.start:%H:%M}   {incident.severity:<9}"
                f"{incident.peak_sigma:>5.1f}σ"
            )
            if st.button(
                label,
                key=f"inc_{incident.id}_{incident.severity.lower()}",
                width="stretch",
                type="primary" if incident.id == st.session_state.selected else "secondary",
            ):
                st.session_state.selected = incident.id
                st.rerun()
    return st.session_state.selected


def incident_detail(incident, scored, p) -> None:
    recurrence = (
        f" · occurrence {incident.occurrence} of this pattern"
        if incident.occurrence > 1
        else ""
    )
    st.markdown(
        f"""<div class="detail-head">
              <span class="id">{incident.id}</span>
              <span class="title">{incident.headline}</span>
            </div>
            <div class="detail-meta">
              <span class="pill {incident.severity.lower()}">{incident.severity}</span>
              &nbsp; {incident.window_label} · {incident.duration_minutes} min ·
              peak {incident.peak_sigma:.1f}σ · {incident.shape}{recurrence}
            </div>""",
        unsafe_allow_html=True,
    )

    evidence = "".join(f"<li>{line}</li>" for line in incident.evidence)
    st.markdown(f'<ul class="evidence">{evidence}</ul>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="hypothesis">
              <span class="label">Hypothesis</span>
              <p>{incident.hypothesis}</p>
            </div>""",
        unsafe_allow_html=True,
    )

    lead_in = max(0, incident.start_index - CONTEXT)
    tail_out = min(len(scored) - 1, incident.end_index + CONTEXT)
    view = scored.iloc[lead_in : tail_out + 1].reset_index(drop=True)
    span = (incident.start_index - lead_in, incident.end_index - lead_in)

    drivers = incident.drivers[:2]
    for column, metric in zip(st.columns(len(drivers)), drivers):
        with column:
            st.plotly_chart(
                theme.metric_chart(view, metric, p, span),
                width="stretch",
                key=f"detail_{incident.id}_{metric}",
            )

    st.plotly_chart(
        theme.contribution_chart(incident, p), width="stretch", key=f"contrib_{incident.id}"
    )

    others = [metric for metric in METRICS if metric not in drivers]
    with st.expander(f"Other metrics in this window ({len(others)})"):
        rest = st.columns(2)
        for index, metric in enumerate(others):
            with rest[index % 2]:
                st.plotly_chart(
                    theme.metric_chart(view, metric, p, span, height=170),
                    width="stretch",
                    key=f"other_{incident.id}_{metric}",
                )


def patterns_and_export(incidents, source: str) -> None:
    left, right = st.columns([1.6, 1])

    with left:
        st.markdown('<p class="section">Recurring patterns</p>', unsafe_allow_html=True)
        patterns = recurring_patterns(incidents)
        if patterns:
            for pattern in patterns:
                st.markdown(
                    f"**{pattern['headline']}** — seen {pattern['count']}× "
                    f"({', '.join(pattern['windows'])})"
                )
        else:
            st.caption(
                "Every incident in this run has a distinct signature, so there is "
                "nothing recurring to report."
            )

    with right:
        st.markdown('<p class="section">Export</p>', unsafe_allow_html=True)
        st.download_button(
            "Incident table (CSV)",
            data=incidents_table(incidents).to_csv(index=False).encode("utf-8"),
            file_name="incidents.csv",
            mime="text/csv",
            width="stretch",
        )
        st.download_button(
            "Investigation report (TXT)",
            data=incident_report(incidents, source).encode("utf-8"),
            file_name="incident_report.txt",
            mime="text/plain",
            width="stretch",
        )

    with st.expander(f"All {len(incidents)} incidents as a table"):
        st.markdown(
            theme.incidents_html(incidents_table(incidents)), unsafe_allow_html=True
        )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
@st.fragment(run_every=0.4)
def replay(raw: pd.DataFrame, threshold: float, p) -> None:
    """Re-detect on a growing window, without blocking the rest of the page.

    Running inside a fragment means only this block reruns on each tick, so
    the sidebar and the controls stay responsive while it plays.
    """
    cursor = st.session_state.get("live_cursor", 60)
    finished = cursor >= len(raw)
    if not finished:
        cursor = min(len(raw), cursor + max(4, len(raw) // 70))
        st.session_state.live_cursor = cursor

    scored, incidents = analyse(raw.iloc[:cursor], threshold)

    st.progress(
        cursor / len(raw),
        text=(
            f"Replay complete — {len(incidents)} incidents over {len(raw)} points"
            if finished
            else f"Streaming · {cursor}/{len(raw)} points · {len(incidents)} incidents so far"
        ),
    )
    summary_strip(incidents, scored)
    st.plotly_chart(
        theme.deviation_chart(scored, threshold, incidents, p),
        width="stretch",
        key="replay_overview",
    )

    columns = st.columns(2)
    for index, metric in enumerate(METRICS):
        with columns[index % 2]:
            st.plotly_chart(
                theme.metric_chart(scored, metric, p, height=200),
                width="stretch",
                key=f"replay_{metric}",
            )

    if incidents:
        st.markdown('<p class="section">Most severe so far</p>', unsafe_allow_html=True)
        for incident in sorted(incidents, key=lambda i: -i.peak_sigma)[:3]:
            st.markdown(
                f"`{incident.window_label}` **{incident.severity}** "
                f"({incident.peak_sigma:.1f}σ) — {incident.headline}"
            )


# ---------------------------------------------------------------------------
def main() -> None:
    if "dark" not in st.session_state:
        st.session_state.dark = _prefers_dark()
    dark = st.session_state.dark
    p = theme.palette(dark)
    st.markdown(theme.css(p, dark), unsafe_allow_html=True)

    frame, source, threshold = sidebar()
    masthead(source, frame)

    if st.session_state.get("live"):
        replay(frame, threshold, p)
        return

    scored, incidents = analyse(frame, threshold)
    summary_strip(incidents, scored)

    if not incidents:
        st.markdown(
            '<div class="empty">No point in this run departs from its own '
            "trailing baseline by the current threshold. Lower the sensitivity "
            "in the sidebar to widen the net.</div>",
            unsafe_allow_html=True,
        )
        return

    rail, detail = st.columns([1, 2.7], gap="large")
    with rail:
        selected_id = incident_rail(incidents)
    with detail:
        incident_detail(next(i for i in incidents if i.id == selected_id), scored, p)

    st.markdown('<p class="section">Whole run</p>', unsafe_allow_html=True)
    st.plotly_chart(
        theme.deviation_chart(scored, threshold, incidents, p, selected_id),
        width="stretch",
        key="overview",
    )

    patterns_and_export(incidents, source)


if __name__ == "__main__":
    main()
