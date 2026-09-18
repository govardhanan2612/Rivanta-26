"""Presentation layer: page styling and the chart builders.

Every chart here plots one series against its baseline, so identity never
rests on colour: the title names the metric, and the only other colour on the
plot is the status red used for flagged points, which always appears beside a
text label elsewhere in the view.
"""

from __future__ import annotations

import plotly.graph_objects as go

from detection import METRIC_LABELS, METRIC_UNITS

# Chart chrome
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
HAIRLINE = "rgba(11,11,11,0.10)"

# Series (slot 1) and the band around it
SERIES = "#2a78d6"
BAND = "rgba(11,11,11,0.05)"

# Status palette, reserved for severity and never used as a series colour
CRITICAL = "#d03b3b"
SERIOUS = "#ec835a"
WARNING = "#fab219"

SEVERITY_COLOR = {"Critical": CRITICAL, "Moderate": SERIOUS, "Minor": WARNING}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
MONO = 'ui-monospace, "Cascadia Mono", "SF Mono", Menlo, monospace'


CSS = f"""
<style>
  [data-testid="stToolbar"], #MainMenu, footer {{ visibility: hidden; height: 0; }}
  [data-testid="stAppViewContainer"] {{ background: {PLANE}; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}

  html, body, [class*="css"] {{ font-family: {FONT}; }}

  /* Masthead */
  .masthead {{ margin-bottom: 1.4rem; }}
  .masthead h1 {{
    font-size: 1.35rem; font-weight: 600; letter-spacing: -0.01em;
    color: {INK}; margin: 0 0 0.15rem 0;
  }}
  .masthead p {{ color: {INK_MUTED}; font-size: 0.82rem; margin: 0; }}

  /* Summary strip */
  .strip {{
    display: flex; gap: 2.4rem; padding: 0.85rem 1.1rem; margin-bottom: 1.4rem;
    background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 8px;
  }}
  .strip .cell {{ display: flex; flex-direction: column; gap: 0.15rem; }}
  .strip .value {{ font-size: 1.15rem; font-weight: 600; color: {INK}; line-height: 1.1; }}
  .strip .label {{
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: {INK_MUTED};
  }}
  .strip .value.critical {{ color: {CRITICAL}; }}
  .strip .value.moderate {{ color: {SERIOUS}; }}

  /* Section headings */
  .section {{
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: {INK_MUTED}; font-weight: 600; margin: 0 0 0.6rem 0;
  }}

  /* Incident rail rows, styled through the stable st-key- class */
  div[class*="st-key-inc_"] button {{
    width: 100%; justify-content: flex-start; text-align: left;
    background: {SURFACE}; color: {INK_SECONDARY};
    border: 1px solid {HAIRLINE}; border-left-width: 3px; border-radius: 6px;
    padding: 0.5rem 0.7rem; margin-bottom: 0.3rem; font-family: {MONO};
    font-size: 0.78rem; font-variant-numeric: tabular-nums; transition: none;
  }}
  div[class*="st-key-inc_"] button p {{
    text-align: left; width: 100%; font-size: 0.78rem; margin: 0;
  }}
  div[class*="st-key-inc_"] button:hover {{ background: #f4f3f0; color: {INK}; }}
  div[class*="st-key-inc_"][class*="_critical"] button {{ border-left-color: {CRITICAL}; }}
  div[class*="st-key-inc_"][class*="_moderate"] button {{ border-left-color: {SERIOUS}; }}
  div[class*="st-key-inc_"][class*="_minor"] button {{ border-left-color: {WARNING}; }}
  div[class*="st-key-inc_"] button[kind="primary"] {{
    background: #eef3fb; color: {INK}; border-color: {SERIES}; border-left-width: 3px;
  }}

  /* Incident detail */
  .detail-head {{ display: flex; align-items: baseline; gap: 0.8rem; margin-bottom: 0.2rem; }}
  .detail-head .id {{
    font-family: {MONO}; font-size: 0.78rem; color: {INK_MUTED};
  }}
  .detail-head .title {{ font-size: 1.05rem; font-weight: 600; color: {INK}; }}
  .detail-meta {{
    font-size: 0.8rem; color: {INK_SECONDARY}; margin-bottom: 1.1rem;
    font-variant-numeric: tabular-nums;
  }}
  .pill {{
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 3px;
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: #fff;
  }}
  .pill.critical {{ background: {CRITICAL}; }}
  .pill.moderate {{ background: {SERIOUS}; }}
  .pill.minor {{ background: {WARNING}; color: {INK}; }}

  .evidence {{ margin: 0 0 1.2rem 0; padding: 0; list-style: none; }}
  .evidence li {{
    font-size: 0.85rem; color: {INK_SECONDARY}; line-height: 1.55;
    padding-left: 0.9rem; position: relative; margin-bottom: 0.3rem;
  }}
  .evidence li:before {{
    content: ""; position: absolute; left: 0; top: 0.62em;
    width: 4px; height: 1px; background: {AXIS};
  }}

  .hypothesis {{
    background: {SURFACE}; border: 1px solid {HAIRLINE};
    border-left: 3px solid {SERIES}; border-radius: 6px;
    padding: 0.8rem 1rem; margin-bottom: 1.2rem;
  }}
  .hypothesis .label {{
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: {INK_MUTED}; font-weight: 600; display: block; margin-bottom: 0.3rem;
  }}
  .hypothesis p {{ font-size: 0.85rem; color: {INK_SECONDARY}; line-height: 1.55; margin: 0; }}

  .empty {{
    background: {SURFACE}; border: 1px dashed {AXIS}; border-radius: 8px;
    padding: 2.4rem; text-align: center; color: {INK_MUTED}; font-size: 0.88rem;
  }}

  [data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {HAIRLINE}; }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
</style>
"""


def _base_layout(height: int, title: str | None = None) -> dict:
    return dict(
        height=height,
        title=(
            dict(text=title, font=dict(size=12, color=INK_SECONDARY), x=0, xanchor="left")
            if title
            else None
        ),
        margin=dict(l=8, r=8, t=30 if title else 8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=11, color=INK_MUTED),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            showgrid=False, showline=True, linecolor=AXIS, linewidth=1,
            ticks="outside", tickcolor=AXIS, ticklen=4, tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
            showline=False, tickfont=dict(size=10),
        ),
    )


def metric_chart(frame, metric: str, span: tuple[int, int] | None = None, height: int = 190):
    """One metric against the band that defines its flagging threshold.

    The shaded region is not a decorative 'normal range' - it is the decision
    boundary itself, so a point outside it is exactly a point the detector
    flagged on this metric.
    """
    fig = go.Figure()
    upper = frame[f"center_{metric}"] + 3 * frame[f"spread_{metric}"]
    lower = frame[f"center_{metric}"] - 3 * frame[f"spread_{metric}"]

    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"], y=upper, mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"], y=lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=BAND, hoverinfo="skip", showlegend=False,
            name="expected band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"], y=frame[metric], mode="lines",
            line=dict(color=SERIES, width=2),
            name=METRIC_LABELS[metric],
            hovertemplate="%{y:.1f} " + METRIC_UNITS[metric] + "<extra></extra>",
        )
    )

    flagged = frame[frame["flagged"]]
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged["timestamp"], y=flagged[metric], mode="markers",
                marker=dict(color=CRITICAL, size=8, line=dict(color=SURFACE, width=2)),
                name="flagged",
                hovertemplate="flagged · %{y:.1f} " + METRIC_UNITS[metric] + "<extra></extra>",
            )
        )

    layout = _base_layout(height, f"{METRIC_LABELS[metric]}  ({METRIC_UNITS[metric]})")
    fig.update_layout(**layout)

    if span:
        start, end = span
        fig.add_vrect(
            x0=frame["timestamp"].iloc[start], x1=frame["timestamp"].iloc[end],
            fillcolor=WARNING, opacity=0.13, line_width=0, layer="below",
        )
    return fig


def deviation_chart(frame, threshold: float, incidents, selected_id: str | None = None):
    """Whole-run overview: how far off baseline the system was, minute by minute."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"], y=frame["deviation"], mode="lines",
            line=dict(color=SERIES, width=1.5), name="peak departure",
            hovertemplate="%{y:.1f}σ<extra></extra>",
        )
    )
    fig.add_hline(y=threshold, line=dict(color=AXIS, width=1))
    fig.add_annotation(
        x=frame["timestamp"].iloc[2], y=threshold, text=f"{threshold:.1f}σ threshold",
        showarrow=False, yshift=9, xanchor="left",
        font=dict(size=9, color=INK_MUTED),
    )

    for incident in incidents:
        is_selected = incident.id == selected_id
        fig.add_vrect(
            x0=frame["timestamp"].iloc[incident.start_index],
            x1=frame["timestamp"].iloc[incident.end_index],
            fillcolor=SEVERITY_COLOR[incident.severity],
            opacity=0.30 if is_selected else 0.13,
            line_width=0, layer="below",
        )

    fig.update_layout(**_base_layout(150, "Peak departure from baseline (σ)"))
    return fig


def contribution_chart(incident, height: int = 150):
    """Share of the incident's deviation owned by each metric."""
    ordered = sorted(incident.contributions.items(), key=lambda item: item[1])
    labels = [METRIC_LABELS[metric] for metric, _ in ordered]
    values = [share * 100 for _, share in ordered]

    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=SERIES, cornerradius=4),
            text=[f"{value:.0f}%" for value in values],
            textposition="outside",
            textfont=dict(size=10, color=INK_SECONDARY),
            hovertemplate="%{x:.1f}% of deviation<extra></extra>",
        )
    )
    layout = _base_layout(height, "Contribution to deviation")
    layout["hovermode"] = "closest"
    layout["xaxis"] = dict(showgrid=False, showline=False, showticklabels=False, range=[0, 118])
    layout["yaxis"] = dict(showgrid=False, showline=False, tickfont=dict(size=10.5, color=INK_SECONDARY))
    layout["bargap"] = 0.45
    fig.update_layout(**layout)
    return fig
