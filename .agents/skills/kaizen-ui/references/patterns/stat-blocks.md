# Stat blocks

Rules for headline-metric tiles (KPI blocks): what every stat block must contain, and how its delta chip is composed and colored. This pattern does not decide grid mechanics or count (see [dashboard.md](../layouts/dashboard.md) → Stat blocks) or chart-type selection ([data-visualization.md](./data-visualization.md)).

## When to use

Grant a stat block only to a metric that answers "are we on track?" at a glance — it's the most visually expensive way to show a number.

| Signal | Treatment |
| --- | --- |
| Headline metric the user checks first; a change would trigger action | Stat block, per [dashboard.md](../layouts/dashboard.md) → Stat blocks count and priority-order rules |
| Metric only interpretable next to peers (per-entity values, per-category splits) | A row in a [ranked list](./ranked-lists.md) — not a tile |
| Static attribute (region, owner, version, type) | A [key-value pair](./key-value-display.md) — never a stat block |

**Default when uncertain:** fewer, bigger. If the user wouldn't act differently when the number changes, it's context, not a KPI.

## Anatomy

Every stat block renders all three, plus a terse label:

1. **Value** — the hero element. Format per [Metric presentation](./data-display.md#metric-presentation).
2. **Comparison** — at least one of: a delta vs. a baseline ("▲ 6.2%"), a share of a whole ("64% of total"), or a distance to a target ("93% of quota").
3. **Scope** — the line that makes the value interpretable: period ("Jun 1 – Jun 30"), denominator ("82% of eligible employees"), or method ("Blended across models").

An optional **sparkline** may sit beneath (see Modifiers).

**Consistency:** every tile in a row uses the same slots in the same order. If one metric has no delta, keep the slot's space or drop the delta from the whole row — don't let tiles diverge in composition.

## Composition rules

### Delta chip: direction and color valence

The delta chip communicates direction (arrow) and magnitude. Color is a claim about goodness — make it deliberately.

| Metric valence | Treatment |
| --- | --- |
| No inherent valence — cost, usage volume, request count: "up" isn't intrinsically good or bad | Neutral chip (gray background, default foreground). The arrow carries direction; color stays out of it. |
| True threshold semantics — error rate, availability vs. SLO, budget vs. cap | Semantic trend color, per [chart color tokens](../design-tokens/CHART_DATA_COLORS.md) → Trend |
| Flat / no change | Muted chip ("▬ 0.4%"), never semantic color |

- **Decide valence once per metric, app-wide.** If cost deltas are neutral on the dashboard, they're neutral on every page — mixed conventions read as meaning where none exists.
- **Never encode direction by color alone.** The arrow glyph is `aria-hidden`; the visible text (and any `title`/accessible name) states direction and magnitude ("up 6.2% vs. previous period") so the chip works without color.
- Every delta names its baseline somewhere in the tile, usually the scope line ("vs. previous period"). A percentage with no baseline is noise.

**Default when uncertain:** neutral. Reserve semantic color for metrics where the team has agreed what "good" means.

### `Badge` is not a delta chip

`Badge` communicates entity status. A delta chip is composed text (arrow + magnitude), not a `Badge` — a `Badge` inside a stat tile implies the metric itself has a workflow state.

## Modifiers

### Sparkline

Include only when the period series exists and the trend's _shape_ adds decision value beyond the delta.

- No axes, labels, or legends — shape and direction only, per [data-visualization.md](./data-visualization.md) → Sparkline.
- Single stroke: `--color-chart-data-categorical-1` by default, or — when the metric also appears as a chart series elsewhere on the page — that series' color, used consistently.
- Sized so it never changes the tile's height.
- Accessibility: `aria-hidden` when the delta chip already states the trend; otherwise `role="img"` with an `aria-label` summarizing it ("cost trending up over 5 weeks").

## Anti-patterns

| Anti-pattern | Why it's wrong | Do this instead |
| --- | --- | --- |
| Bare value ("$4.82M" alone) | The user can't answer "so what?" — no baseline, no scope | Render value + comparison + scope |
| Semantic green/red on a valence-free metric | Colors an increase "bad" (or "good") without the team having decided that | Neutral delta chip; semantic color only for true thresholds |
| Direction shown by color alone | Fails color-blind users and grayscale contexts | `aria-hidden` arrow + visible magnitude text |
| Tiles with different compositions in one row | Row reads as unrelated boxes; eyes re-parse each tile | Same slots, same order, every tile |
| Sparkline with axes or point labels | It's a chart now — competes with the value | Shape-only sparkline, or promote to a real chart panel |
| Delta with no baseline named anywhere | "▲ 12.4%" against what? | Name the baseline in the scope line or the chip's accessible name |
