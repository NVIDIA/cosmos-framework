# Data visualization

Choosing and coloring a chart. The chart type must serve the analytical task the user is performing, not just display numbers — the most common failure is picking the type that "looks like a dashboard" instead of the one the data shape requires.

## When to use

Render a chart when the prompt or data shape implies comparison, trend, distribution, or proportion:

- The prompt names a chart, or the data compares values across categories
- The data shows change over time, or a part-to-whole breakdown
- The data groups values by a shared dimension (team, region, status)

Don't use when:

- The value is a single scalar with no comparable dimension and the prompt doesn't ask for a chart — use a stat tile or inline text instead
- The user needs exact values, many attributes per item, or to sort/filter — a [`Table`](./data-display.md) serves precision better than a chart
- The chart would sit on a dashboard — read [dashboard.md](../layouts/dashboard.md) first for stat-block placement, chart-grid layout, and per-dashboard micro-rules (gridlines, hero chart spanning) before applying this file's chart-type and color rules inside each chart

## Variants

Head each variant with the analytical task, not the chart's name.

| Situation | Chart | Notes |
| --- | --- | --- |
| Comparing 2–20 discrete categories | Bar | Horizontal when labels are long or there are 7+ categories. Stacked when sub-segments compose a total; grouped for 2–4 side-by-side sub-categories. |
| Change over time, continuous/sequential x-axis | Line | **Default for "trend" titles** — a title containing "trend," "over time," "progression," or a time range is a time series; do not default to a bar chart for it. |
| Volume/magnitude over time, or parts-of-a-whole over time | Area | Use sparingly — overlapping series obscure each other. Stacked area works best with 2–4 series. |
| Parts of a whole, 2–5 segments | Pie / donut | 6+ segments: use a horizontal bar chart instead — slices become indistinguishable. Prefer donut when a summary metric can sit in the center (value + one-word caption, `tabular-nums`); its legend sits beside it as label / value / percent rows, not bare swatches. |
| A single value's position in a defined range with meaningful thresholds | [`Meter`](../components/Meter.md) / [`SegmentedMeter`](../components/SegmentedMeter.md) | Not for comparing multiple values — that's a bar chart. |
| Compact inline trend (table row, card, stat block) | Sparkline | No axes, labels, or legends — shape and direction only. |
| Density/intensity across two dimensions | Heatmap | Large datasets where aggregate pattern matters more than individual points. |
| Relationship between two numeric variables | Scatter | Each point is an individual item, not an aggregate. |

**Bar vs. line is the highest-stakes call.** Bars imply discrete categorical comparison; lines imply continuity between points. A "Utilization trend" or "Daily active users" chart is a time series — line or area, never bars.

## Composition rules

Apply to every chart, dashboard or not.

### Tooltips are required, per-point labels usually aren't

Every chart shows a tooltip on hover/tap with the point's exact value, its label/category, and any context that makes it interpretable (date, percent of total). A small number of **selective callouts** (peak, latest point, an anomaly — 2–3 max) is fine on top of tooltips; labeling every point is the decluttering violation tooltips exist to prevent.

### Legends

Include one whenever more than one series/color is in play — render a real [`Legend`](../components/Legend.md) next to the chart rather than inventing swatches, and keep entries in the same order as the series they describe (`Legend`'s own guidance covers pinning `color` when order can change). Place it above or beside the chart, never below — below competes with axis labels.

### Axis labels

Label both axes, name the unit in the label ("Response Time (ms)", not "Response Time"), and keep tick intervals readable rather than dense.

### Text alternative (accessibility)

A chart rendered as SVG is invisible to a screen reader — its takeaway must also exist as text:

- Give the chart an `aria-label` (or `<title>`/`<desc>`) stating the takeaway, not just the name — "Cost by week, trending up ~18% over 6 weeks," not "Cost chart."
- Make the underlying numbers reachable as data — a "View as table" toggle, a visually-hidden `<table>`, or the row's drill destination. A tooltip is hover-only and keyboard-inaccessible, so it does not satisfy this.

## Modifiers

### Color

See [chart and data-visualization color tokens](../design-tokens/CHART_DATA_COLORS.md) for the token families and when each applies. In short: categorical for unrelated series/segments, sequential for a single quantity's intensity, trend tokens only when a value has a favorable/unfavorable direction. Never use trend (semantic) colors for arbitrary categorical differentiation, and never rely on color alone to carry meaning — pair it with a label, pattern, or shape, and keep marker/line contrast ≥ 3:1 and any label text ≥ 4.5:1 against the panel background.

## Anti-patterns

| Anti-pattern | Why it's wrong | Do this instead |
| --- | --- | --- |
| Bar chart for a time-series title ("Utilization trend," "Daily active users") | Bars read as discrete unordered categories, hiding the continuity the user actually needs to see | Line chart, or area for cumulative volume |
| Pie/donut with 6+ segments | Slices become too small to differentiate, and precise comparison is impossible | Horizontal bar chart |
| Labeling every point/bar when tooltips already exist | Duplicates the tooltip and clutters the chart | Tooltips plus at most 2–3 selective callouts (peak/latest/anomaly) |
| Shipping a chart with the charting library's default white background and black axis text in a dark-themed app | Reads as visibly unfinished and fails contrast against the surrounding surface | Theme the chart explicitly — background, text, gridlines, tooltip all pulled from app tokens |
| A chart with no text alternative | Invisible to screen readers — the insight the chart carries doesn't reach every user | `aria-label` stating the takeaway, plus the underlying values reachable as a table |
