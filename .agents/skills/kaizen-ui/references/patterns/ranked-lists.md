# Ranked lists

Rules for ranked-contribution displays — "top N by X" (models by cost, departments by spend, endpoints by error volume). Decides when a ranked list beats a table, chart, or donut; row anatomy; bar color; and drill interaction. Does not decide chart types in general ([data-visualization.md](./data-visualization.md)) or full entity tables with sorting/filtering ([data-display.md](./data-display.md)).

## When to use

Users are scanning for "who or what contributes most" — pick the display that fits the data shape.

| Data shape | Use |
| --- | --- |
| 5–20 items, one primary measure, share-of-whole matters, scanning intent | Ranked list (this pattern) |
| ≤ 5 parts that compose a whole | Stacked bar or donut, [data-visualization.md](./data-visualization.md) |
| Users need several attributes per item, or sort/filter/search | A [`Table`](../components/Table.md) — a ranked list has exactly one ranking |
| Precise magnitude comparison against an axis | A horizontal bar chart in a chart Panel, [data-visualization.md](./data-visualization.md) |
| A single scalar with no comparable dimension | A [stat block](./stat-blocks.md) — not a one-row list |

**Default when uncertain:** ranked list for "top contributors" panels on dashboards. It degrades gracefully — remove the bars and it's still a legible label/value list.

## Anatomy

Base row anatomy (label + right-aligned value) is [dashboard.md](../layouts/dashboard.md) → Ranking list panel. This pattern extends it with:

1. **Series dot** (optional) — only when rows correspond to chart series shown elsewhere (see Composition rules).
2. **Share bar** — proportional fill in a fixed-width track; the largest row (or the whole, when shares sum to it) sets 100%.
3. **Value + share columns** — right-aligned, `tabular-nums`, formatted per [Metric presentation](./data-display.md#metric-presentation).
4. **Delta / status** (optional) — a period delta per [stat-blocks.md](./stat-blocks.md) valence rules, or an informational `Badge` (`color="blue"`) for a genuinely new entrant.

**Rules:**

- **Row ceiling:** top 5–8 rows plus one muted, aggregated **"Other (N)"** row so shares visibly sum to the whole. Never render every contributor — link "View all" to the full table.
- Left-align the label column, right-align every numeric column; one row height, no wrapping.
- Label and value are separate elements, never a single concatenated text run ("DGX Cloud $1.12M").

## Composition rules

### Bar and dot color

| Situation | Treatment |
| --- | --- |
| Rows correspond to chart series shown elsewhere on the page (e.g. models that also appear in a trend chart) | Same series colors with the same assignment everywhere, per [chart color tokens](../design-tokens/CHART_DATA_COLORS.md) → Categorical — include the series dot |
| Rows are not chart series anywhere | Single accent (or neutral) bars — rank order already carries the signal; per-row colors add noise |
| "Other" aggregate row | Always neutral/muted, regardless of the scheme above |
| New entrant | `Badge color="blue"` ("New") — never a unique bar color |

### Interaction and drill

- **Row navigation:** when each row is an entity with a destination, the whole row is the click target — no per-row "View" buttons.
- Clicking a row and clicking a panel-level "View all" link may go to different depths (entity vs. full list) — that's correct progressive disclosure, not duplication.
- A ranked list with no destination for its rows is fine — just don't invent a page for it.

## Anti-patterns

| Anti-pattern | Why it's wrong | Do this instead |
| --- | --- | --- |
| Scattering label and stats across the panel width instead of fixed columns | Values float away from labels; the value column stops being scannable | Fixed columns: label (flexible) + bar + right-aligned numerics |
| Per-row rainbow bars with no series meaning | Color implies categories that don't exist | Single accent bars; series colors only when rows are chart series |
| Rendering every contributor (20, 40 rows) in the panel | The panel becomes an unpaginated table, losing its summary role | Top 5–8 + muted "Other (N)" + "View all" |
| Omitting the "Other" bucket when showing shares | Percentages visibly don't sum; users distrust the data | Always aggregate the tail when truncating a share-based list |
| Values as Tags/Badges/pills in each row | Pills compound into noise in dense rows, per [data-display.md](./data-display.md) → Taxonomic labels | Plain text values; one optional delta chip or "New" `Badge` |
| Sortable headers on a ranked-list panel | It's now an under-featured table | Keep the fixed ranking; link to the real table for sorting |
