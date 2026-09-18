"""Presentation layer: palettes, page styling and the chart builders.

Both modes are selected rather than flipped. The dark steps are chosen for the
dark surface - an inverted light palette loses its contrast relationships and
reads muddy - while the status colours are deliberately identical in both, so
that severity never changes meaning with the theme.

Every chart plots one series against its baseline, so identity never rests on
colour: the title names the metric, and the only other colour on the plot is
the status red used for flagged points, which always appears beside a text
label elsewhere in the view.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from detection import METRIC_LABELS, METRIC_UNITS


@dataclass(frozen=True)
class Palette:
    surface: str       # card / chart surface
    plane: str         # page behind the cards
    ink: str           # primary text
    ink_secondary: str # body text
    ink_muted: str     # axis labels, captions
    grid: str          # hairline gridlines
    axis: str          # baseline and axis rules
    hairline: str      # card borders
    series: str        # categorical slot 1
    band: str          # the decision-boundary fill
    hover: str         # rail row hover
    selected: str      # rail row selected
    shadow: str


LIGHT = Palette(
    surface="#fcfcfb",
    plane="#f9f9f7",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    hairline="rgba(11,11,11,0.10)",
    series="#2a78d6",
    band="rgba(11,11,11,0.05)",
    hover="#f4f3f0",
    selected="#eef3fb",
    shadow="rgba(11,11,11,0.04)",
)

DARK = Palette(
    surface="#1a1a19",
    plane="#0d0d0d",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    hairline="rgba(255,255,255,0.10)",
    series="#3987e5",
    band="rgba(255,255,255,0.06)",
    hover="#242422",
    selected="#16283f",
    shadow="rgba(0,0,0,0.30)",
)

# Status palette is fixed, never themed. All four clear 3:1 against both
# surfaces, so severity reads the same whichever mode you are in.
CRITICAL = "#d03b3b"
SERIOUS = "#ec835a"
WARNING = "#fab219"
SEVERITY_COLOR = {"Critical": CRITICAL, "Moderate": SERIOUS, "Minor": WARNING}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
MONO = 'ui-monospace, "Cascadia Mono", "SF Mono", Menlo, monospace'


def palette(dark: bool) -> Palette:
    return DARK if dark else LIGHT


def css(p: Palette, dark: bool) -> str:
    """Full stylesheet for one mode.

    Streamlit resolves its own theme once at startup from config.toml, so a
    runtime switch has to restate the chrome as well as our own components.
    """
    minor_ink = p.ink if dark else "#0b0b0b"
    return f"""
<style>
  [data-testid="stToolbar"], #MainMenu, footer {{ visibility: hidden; height: 0; }}
  [data-testid="stAppViewContainer"], [data-testid="stMain"],
  section[data-testid="stMain"] > div {{ background: {p.plane}; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
  html, body, [class*="css"] {{ font-family: {FONT}; }}

  /* Streamlit chrome, restated for the active mode */
  [data-testid="stSidebar"] {{
    background: {p.surface}; border-right: 1px solid {p.hairline};
  }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] span, [data-testid="stSidebar"] li {{
    color: {p.ink_secondary};
  }}
  [data-testid="stSidebar"] strong {{ color: {p.ink}; }}
  [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"],
  [data-testid="stCaptionContainer"] p {{ color: {p.ink_muted} !important; }}
  .stMarkdown p, .stMarkdown li {{ color: {p.ink_secondary}; }}
  .stMarkdown strong {{ color: {p.ink}; }}
  .stMarkdown code {{
    background: {p.hover}; color: {p.ink_secondary};
    border: 1px solid {p.hairline}; border-radius: 3px; padding: 0.05rem 0.3rem;
  }}
  [data-testid="stExpander"] {{
    background: {p.surface}; border: 1px solid {p.hairline}; border-radius: 8px;
  }}
  [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {{
    color: {p.ink_secondary};
  }}
  [data-testid="stFileUploaderDropzone"] {{
    background: {p.plane}; border: 1px dashed {p.axis};
  }}
  [data-testid="stFileUploaderDropzone"] span,
  [data-testid="stFileUploaderDropzone"] small {{ color: {p.ink_muted}; }}
  [data-testid="stProgress"] p {{ color: {p.ink_muted}; }}
  [data-baseweb="slider"] div[role="slider"] ~ div {{ color: {p.ink_secondary}; }}

  /* Ordinary buttons. Streamlit paints these from the theme it resolved at
     startup, so without this they stay light behind a runtime switch. The
     incident rail overrides these below on higher specificity. */
  .stButton button, .stDownloadButton button, [data-testid="stBaseButton-secondary"],
  [data-testid="stFileUploaderDropzone"] button {{
    background: {p.surface} !important; color: {p.ink_secondary} !important;
    border: 1px solid {p.hairline} !important; border-radius: 6px;
    font-weight: 500;
  }}
  .stButton button:hover, .stDownloadButton button:hover,
  [data-testid="stFileUploaderDropzone"] button:hover {{
    background: {p.hover} !important; color: {p.ink} !important;
    border-color: {p.series} !important;
  }}
  .stButton button p, .stDownloadButton button p,
  .stButton button div, .stDownloadButton button div {{ color: inherit !important; }}
  .stButton button:focus, .stDownloadButton button:focus {{ box-shadow: none !important; }}

  /* Masthead */
  .masthead {{ margin-bottom: 1.4rem; }}
  .masthead h1 {{
    font-size: 1.35rem; font-weight: 600; letter-spacing: -0.01em;
    color: {p.ink}; margin: 0 0 0.15rem 0;
  }}
  .masthead p {{ color: {p.ink_muted}; font-size: 0.82rem; margin: 0; }}

  /* Summary strip */
  .strip {{
    display: flex; gap: 2.4rem; padding: 0.85rem 1.1rem; margin-bottom: 1.4rem;
    background: {p.surface}; border: 1px solid {p.hairline}; border-radius: 8px;
  }}
  .strip .cell {{ display: flex; flex-direction: column; gap: 0.15rem; }}
  .strip .value {{ font-size: 1.15rem; font-weight: 600; color: {p.ink}; line-height: 1.1; }}
  .strip .label {{
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: {p.ink_muted};
  }}
  .strip .value.critical {{ color: {CRITICAL}; }}
  .strip .value.moderate {{ color: {SERIOUS}; }}

  .section {{
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: {p.ink_muted}; font-weight: 600; margin: 0 0 0.6rem 0;
  }}

  /* Incident rail rows, styled through the stable st-key- class */
  div[class*="st-key-inc_"] button {{
    width: 100%; justify-content: flex-start !important; text-align: left;
    background: {p.surface} !important; color: {p.ink_secondary} !important;
    border: 1px solid {p.hairline} !important; border-left-width: 3px !important;
    border-radius: 6px;
    padding: 0.5rem 0.7rem; margin-bottom: 0.3rem; font-family: {MONO};
    font-size: 0.78rem; font-variant-numeric: tabular-nums; transition: none;
  }}
  div[class*="st-key-inc_"] button p {{
    text-align: left; width: 100%; font-size: 0.78rem; margin: 0;
    color: inherit !important; font-family: {MONO};
  }}
  div[class*="st-key-inc_"] button:hover {{
    background: {p.hover} !important; color: {p.ink} !important;
  }}
  div[class*="st-key-inc_"][class*="_critical"] button {{ border-left-color: {CRITICAL} !important; }}
  div[class*="st-key-inc_"][class*="_moderate"] button {{ border-left-color: {SERIOUS} !important; }}
  div[class*="st-key-inc_"][class*="_minor"] button {{ border-left-color: {WARNING} !important; }}
  div[class*="st-key-inc_"] button[kind="primary"] {{
    background: {p.selected} !important; color: {p.ink} !important;
    border-color: {p.series} !important; border-left-width: 3px !important;
  }}
  div[class*="st-key-inc_"] button[kind="primary"] p {{ color: {p.ink} !important; }}

  /* Incident detail */
  .detail-head {{ display: flex; align-items: baseline; gap: 0.8rem; margin-bottom: 0.2rem; }}
  .detail-head .id {{ font-family: {MONO}; font-size: 0.78rem; color: {p.ink_muted}; }}
  .detail-head .title {{ font-size: 1.05rem; font-weight: 600; color: {p.ink}; }}
  .detail-meta {{
    font-size: 0.8rem; color: {p.ink_secondary}; margin-bottom: 1.1rem;
    font-variant-numeric: tabular-nums;
  }}
  .pill {{
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 3px;
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: #fff;
  }}
  .pill.critical {{ background: {CRITICAL}; }}
  .pill.moderate {{ background: {SERIOUS}; color: #0b0b0b; }}
  .pill.minor {{ background: {WARNING}; color: #0b0b0b; }}

  .evidence {{ margin: 0 0 1.2rem 0; padding: 0; list-style: none; }}
  .evidence li {{
    font-size: 0.85rem; color: {p.ink_secondary}; line-height: 1.55;
    padding-left: 0.9rem; position: relative; margin-bottom: 0.3rem;
  }}
  .evidence li:before {{
    content: ""; position: absolute; left: 0; top: 0.62em;
    width: 4px; height: 1px; background: {p.axis};
  }}

  .hypothesis {{
    background: {p.surface}; border: 1px solid {p.hairline};
    border-left: 3px solid {p.series}; border-radius: 6px;
    padding: 0.8rem 1rem; margin-bottom: 1.2rem;
  }}
  .hypothesis .label {{
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: {p.ink_muted}; font-weight: 600; display: block; margin-bottom: 0.3rem;
  }}
  .hypothesis p {{ font-size: 0.85rem; color: {p.ink_secondary}; line-height: 1.55; margin: 0; }}

  .empty {{
    background: {p.surface}; border: 1px dashed {p.axis}; border-radius: 8px;
    padding: 2.4rem; text-align: center; color: {p.ink_muted}; font-size: 0.88rem;
  }}

  /* The incident table is hand-rolled rather than st.dataframe, which paints
     to a canvas using the theme Streamlit resolved at startup and so would
     stay light behind a runtime switch. */
  table.incidents {{
    width: 100%; border-collapse: collapse; font-size: 0.78rem;
    font-variant-numeric: tabular-nums; color: {p.ink_secondary};
  }}
  table.incidents th {{
    text-align: left; font-weight: 600; color: {p.ink_muted}; font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 0.05em;
    padding: 0.4rem 0.7rem 0.4rem 0; border-bottom: 1px solid {p.axis};
    white-space: nowrap;
  }}
  table.incidents td {{
    padding: 0.45rem 0.7rem 0.45rem 0; border-bottom: 1px solid {p.grid};
    vertical-align: top;
  }}
  table.incidents tr:last-child td {{ border-bottom: none; }}
  table.incidents td.sev-critical {{ color: {CRITICAL}; font-weight: 600; }}
  table.incidents td.sev-moderate {{ color: {SERIOUS}; font-weight: 600; }}
  table.incidents td.sev-minor {{ color: {minor_ink}; font-weight: 600; }}
</style>
"""


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _base_layout(p: Palette, height: int, title: str | None = None) -> dict:
    return dict(
        height=height,
        title=(
            dict(text=title, font=dict(size=12, color=p.ink_secondary), x=0, xanchor="left")
            if title
            else None
        ),
        margin=dict(l=8, r=8, t=30 if title else 8, b=8),
        paper_bgcolor=p.surface,
        plot_bgcolor=p.surface,
        font=dict(family=FONT, size=11, color=p.ink_muted),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=p.surface,
            bordercolor=p.axis,
            font=dict(family=FONT, size=11, color=p.ink),
        ),
        xaxis=dict(
            showgrid=False, showline=True, linecolor=p.axis, linewidth=1,
            ticks="outside", tickcolor=p.axis, ticklen=4, tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=p.grid, gridwidth=1, zeroline=False,
            showline=False, tickfont=dict(size=10),
        ),
    )


def metric_chart(frame, metric: str, p: Palette, span=None, height: int = 190):
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
            fill="tonexty", fillcolor=p.band, hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"], y=frame[metric], mode="lines",
            line=dict(color=p.series, width=2), name=METRIC_LABELS[metric],
            hovertemplate="%{y:.1f} " + METRIC_UNITS[metric] + "<extra></extra>",
        )
    )

    flagged = frame[frame["flagged"]]
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged["timestamp"], y=flagged[metric], mode="markers",
                marker=dict(color=CRITICAL, size=8, line=dict(color=p.surface, width=2)),
                name="flagged",
                hovertemplate="flagged · %{y:.1f} " + METRIC_UNITS[metric] + "<extra></extra>",
            )
        )

    fig.update_layout(**_base_layout(p, height, f"{METRIC_LABELS[metric]}  ({METRIC_UNITS[metric]})"))

    if span:
        start, end = span
        fig.add_vrect(
            x0=frame["timestamp"].iloc[start], x1=frame["timestamp"].iloc[end],
            fillcolor=WARNING, opacity=0.16, line_width=0, layer="below",
        )
    return fig


def deviation_chart(frame, threshold: float, incidents, p: Palette, selected_id=None):
    """Whole-run overview: how far off baseline the system was, minute by minute."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"], y=frame["deviation"], mode="lines",
            line=dict(color=p.series, width=1.5),
            hovertemplate="%{y:.1f}σ<extra></extra>",
        )
    )
    fig.add_hline(y=threshold, line=dict(color=p.axis, width=1))
    fig.add_annotation(
        x=frame["timestamp"].iloc[2], y=threshold, text=f"{threshold:.1f}σ threshold",
        showarrow=False, yshift=9, xanchor="left", font=dict(size=9, color=p.ink_muted),
    )

    for incident in incidents:
        fig.add_vrect(
            x0=frame["timestamp"].iloc[incident.start_index],
            x1=frame["timestamp"].iloc[incident.end_index],
            fillcolor=SEVERITY_COLOR[incident.severity],
            opacity=0.34 if incident.id == selected_id else 0.15,
            line_width=0, layer="below",
        )

    fig.update_layout(**_base_layout(p, 150, "Peak departure from baseline (σ)"))
    return fig


def contribution_chart(incident, p: Palette, height: int = 150):
    """Share of the incident's deviation owned by each metric."""
    ordered = sorted(incident.contributions.items(), key=lambda item: item[1])
    labels = [METRIC_LABELS[metric] for metric, _ in ordered]
    values = [share * 100 for _, share in ordered]

    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=p.series, cornerradius=4),
            text=[f"{value:.0f}%" for value in values],
            textposition="outside", textfont=dict(size=10, color=p.ink_secondary),
            hovertemplate="%{x:.1f}% of deviation<extra></extra>",
        )
    )
    layout = _base_layout(p, height, "Contribution to deviation")
    layout["hovermode"] = "closest"
    layout["xaxis"] = dict(showgrid=False, showline=False, showticklabels=False, range=[0, 118])
    layout["yaxis"] = dict(
        showgrid=False, showline=False, tickfont=dict(size=10.5, color=p.ink_secondary)
    )
    layout["bargap"] = 0.45
    fig.update_layout(**layout)
    return fig


def incidents_html(table: pd.DataFrame) -> str:
    """The incident table as themed HTML, with severity carrying a class."""
    header = "".join(f"<th>{column}</th>" for column in table.columns)
    rows = []
    for _, row in table.iterrows():
        cells = []
        for column in table.columns:
            klass = (
                f' class="sev-{str(row[column]).lower()}"' if column == "Severity" else ""
            )
            cells.append(f"<td{klass}>{row[column]}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<table class="incidents"><thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
