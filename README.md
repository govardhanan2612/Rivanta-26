# Incident triage

Anomaly detection and root cause analysis over system metrics. The app ingests
a stream of server metrics — CPU, memory, response time, network throughput —
finds the moments the system left its own normal behaviour, groups them into
incidents, and explains each one from measurements rather than from a rule
about which metric moved.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Runs on synthetic data out of the box. Upload a CSV with `timestamp` plus the
four metric columns to run it on your own.

## What it does

The screen is a triage queue, not a dashboard. Incidents are ranked by
severity on the left; selecting one opens its evidence, the metrics that drove
it, and the window it occupied. The point is to answer *what happened and
why*, not to show four line charts and leave the reading to you.

For each incident the app reports:

- **When it started**, distinguished from when it tripped the alarm. For
  anything gradual these differ, and the earlier time is the useful one.
- **Which metrics drove it**, as a share of the total deviation, with the
  requirement that a metric be genuinely displaced before it is named — not
  merely the least quiet one in a calm window.
- **What shape it had** — arrived at once, or accumulated.
- **Whether the drivers moved together**, as a correlation between their
  departures.
- **A hypothesis**, branched on the measured geometry, with the measurements
  it rests on printed beside it.

## How detection works

Two signals over the same data.

**Trailing baseline (primary).** Each metric is scored against a rolling
median of its own recent past, with spread from a median absolute deviation.
MAD rather than standard deviation, because the anomalies we are hunting would
otherwise inflate the very baseline used to find them. The window is shifted by
one sample so a point never contributes to the baseline it is judged against —
without that, sustained faults quietly hide themselves.

The result is in sigma, an absolute unit. That matters: a severity that is
min-max scaled against the current window changes as new data arrives, so the
same incident can slide from Critical to Minor while you are looking at it.

**Isolation Forest (corroborating).** Fitted on the z-scores rather than raw
values, which makes it scale-free across metrics and gives it the same temporal
context the primary signal has. Its job is the points that look unremarkable
metric by metric but sit somewhere odd once the metrics are considered jointly.
It runs with `contamination="auto"`, so it is not forced to flag a set fraction.

A point is flagged on a clear single-metric departure, or on a slightly milder
one that the forest also considers structurally odd. The corroborated branch
reaches only a little below the threshold on purpose — any lower and it admits
ordinary noise, which no amount of downstream grouping can undo.

**Persistence.** A crossing on a single sample is what noise looks like at any
sensible sigma; across four metrics and several hundred samples, some points
cross by arithmetic alone. An incident must survive consecutive samples unless
it is extreme enough to stand on one. This is the standard alerting remedy and
it is what keeps a clean run genuinely empty rather than merely quieter.

## Why root cause is measured, not looked up

The obvious way to write this is a table from metric names to conclusions —
if memory and response time are both high, say "memory leak". That is a
hardcoded answer wearing an analysis costume, and it cannot be right about
anything it was not told in advance.

Instead each incident is measured: contribution shares, the largest single-step
jump against the total departure, a correlation between the drivers' residuals.
The hypothesis branches on *that* geometry — a fault that accumulates across
many samples while two metrics climb in step is resource exhaustion whichever
metrics they happen to be. Metric names only colour the wording once a branch
is chosen.

Shape and correlation are measured from onset, against a reference frozen just
before it. Measuring against the live trailing baseline is self-defeating:
during a slow climb the baseline climbs too, so a genuine ramp reads flat.

Recurring patterns are grouped by measured signature rather than clustered. With
a handful of incidents in a window, k-means is fitting noise, and "this is the
third time" is more use to an operator than a cluster id.

## Measured behaviour

Parameters were chosen by sweeping them against seeded clean and faulty runs
rather than by taste (`threshold` × persistence, 12 seeds each):

| | |
|---|---|
| Recall of injected faults | 6/6, at every threshold tested |
| False incidents, clean data | 0.17 per 500-point run |
| False incidents, faulty run | 0 — six incidents, six faults |
| Gradual fault caught | 3 samples into a 14-sample ramp |

The faults injected into the synthetic data differ in shape, not just size, so
the detector has to do more than find large numbers: a CPU spike, a mild
sustained elevation, a two-metric gradual climb, a throughput collapse, a sharp
latency spike, and a moderate step.

The sensitivity slider is an absolute cut, so lowering it widens the net and
raising it past every departure correctly returns nothing at all.

## Layout

| File | |
|---|---|
| `app.py` | Layout and wiring |
| `detection.py` | Simulation, baseline, the two detection signals |
| `incidents.py` | Grouping, measurement, explanation |
| `theme.py` | Styling and chart builders |
