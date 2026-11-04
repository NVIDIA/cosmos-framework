# Chart and data-visualization color tokens

Theme-aware color tokens for series, segments, gradients, and chart chrome. **Default to these over inventing a color** — they follow the active theme, and choosing the right family is what makes a chart readable rather than merely colorful.

Pick the family from the **relationship between the values**, not from how the color looks:

| The values are                                     | Family                      |
| -------------------------------------------------- | --------------------------- |
| Unrelated categories or series, no inherent order  | [Categorical](#categorical) |
| One quantity varying in intensity across a range   | [Sequential](#sequential)   |
| Carrying a favorable or unfavorable direction      | [Trend](#trend)             |
| Chart chrome — gridlines, axis strokes, tick lines | [Structural](#structural)   |

## Applying a token

| Need                                       | Spelling                                                           |
| ------------------------------------------ | ------------------------------------------------------------------ |
| A CSS variable, or any prop taking a color | `var(--color-chart-data-categorical-1)`                            |
| An SVG mark — the usual case for a chart   | `fill-chart-data-categorical-1`, `stroke-chart-data-categorical-1` |
| A swatch background or label text          | `bg-chart-data-categorical-1`, `text-chart-data-categorical-1`     |

SVG `fill` and `stroke` accept `var()` directly, so an SVG-based charting library can take the token string as-is. A canvas-based library cannot: resolve it first with `getComputedStyle(element).getPropertyValue("--color-chart-data-categorical-1")`, and re-resolve on theme change, since the value differs between light and dark.

## Categorical

Use to distinguish unrelated categories or series — chart series, segments in a stacked or segmented visualization — not for values with a good/bad meaning. Assign by position, and keep the same category-to-color assignment across every chart on the same page.

`--color-chart-data-categorical-1` through `--color-chart-data-categorical-8`.

[Legend](../components/Legend.md) entries and [SegmentedMeter](../components/SegmentedMeter.md) segments already fall back to this scale by position, so two lists of the same length line up without any color wiring — and can drift apart if one list reorders.

More than 8 categories in one chart is itself a signal to reduce the category count — group a long tail into "Other" — rather than to extend the palette. Distinct hues stop being distinguishable past 8, and the scale cycles rather than growing.

## Sequential

Use for a single quantity that varies in intensity or density across a continuous range, such as a heatmap cell or a gradient fill — not for distinct categories.

Two hue ramps, each low to high: `--color-chart-data-sequential-blue-1` through `-7`, and `--color-chart-data-sequential-green-1` through `-7`. Pick blue for a neutral quantity; pick green only when "more" is already the desired reading, since it otherwise competes with the trend tokens below.

## Trend

Use only when the value has a meaningful favorable or unfavorable direction, such as a KPI delta or a threshold-based reading — never for arbitrary categorical differentiation. Choose the direction from the outcome rather than from whether the number rose or fell: falling latency is a positive trend.

| Token                                     | Use for                                      |
| ----------------------------------------- | -------------------------------------------- |
| `--color-chart-data-trend-positive-1/2/3` | A favorable reading, low to high emphasis    |
| `--color-chart-data-trend-negative-1/2/3` | An unfavorable reading, low to high emphasis |
| `--color-chart-data-trend-neutral`        | Unchanged, or no meaningful direction        |

This is a separate family from the [feedback tokens](./COLORS.md#feedback): feedback tokens are for status messaging UI such as banners and badges, trend tokens are for chart marks and deltas.

## Structural

Use `border-chart` or `var(--border-color-chart)` for chart borders, gridlines, axis strokes, and tick lines.
