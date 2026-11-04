# Data Display

Rules for presenting collections of structured data.

Use when:

- Presenting a collection of structured data items
- Users need to scan, compare, filter, sort, or select items
- Each item has identifiable attributes (title, status, metadata)

Don't use when:

- Content is free-form prose or narrative — use typography/layout instead
- Displaying a single value with no comparison context — use a `Badge` or inline text
- Showing chronological events — use an [activity feed pattern](./activity-feed.md)
- Rendering key-value metadata pairs only — use a [key-value display pattern](./key-value-display.md)

---

## Variants

The central decision: **table vs cards vs list**. Choose based on content shape and user task.

| Variant | Use when | Avoid when |
| --- | --- | --- |
| **Table** | Data has consistent structure across items (same fields per row); users compare values across items; sorting/filtering by specific columns is needed; data is dense and benefits from tight vertical spacing; actions apply to individual rows (edit, delete, view) | Items have highly variable content; visual elements (images, charts) are important; mobile is a primary use case (tables don't adapt well) |
| **Cards** | Items are "objects" users browse or explore; each item has visual elements (images, icons, status indicators); content length varies significantly between items; users benefit from seeing more context per item; discovery or comparison is the goal | Users need to compare specific values across many items; sorting by multiple dimensions is important; screen space is limited and density matters |
| **List** | Items are simple (1–2 lines of content); users scan sequentially to find something; items represent actions, events, or navigation targets; vertical space is limited | Items have multiple attributes to display; comparison between items matters |

**Related anatomy specs:**

- For card content slots / order / typography → [entity-cards.md](./entity-cards.md)

---

## Composition rules

Apply across all variants.

### Taxonomic labels in table cells: plain text, not pills

When a table cell holds multiple categorical values (e.g., a "Teams" column where each row has 2–4 team names, a "Tags" column with tech stack labels), **do not render each value as a `Tag` or `Badge`**. In dense data tables, pills compound into visual noise.

**Pattern:** render the values as plain text or navigable `Anchor` components, separated by commas or middle-dots.

**When `Tag` IS appropriate in a table:**

- A single-value category column where every row has exactly one Tag (e.g., a "Type" column: Model / Blueprint / Service) — one pill per row doesn't compound

**Why:** A scanning-optimized table prioritizes row-to-row comparison. Pills interrupt the visual row by breaking the text baseline and introducing color noise. Plain text preserves the scanning rhythm; `Anchor` adds affordance without visual weight.

### Badge variant selection

Use `kind="solid"` for status badges in **detail views, side panels, and status bars** — the solid variant provides clear, immediate visual hierarchy for status. The outline variant is visually distracting when it appears alongside other bordered elements (panels, inputs, dividers).

Use `kind="outline"` only when the badge is **secondary information** that should not compete for attention — for example, metadata tags in a table cell or labels in a dense list where multiple badges appear per row.

| Context | Badge kind | Reason |
| --- | --- | --- |
| Side panel status | `solid` | Primary signal for triage; needs to stand out |
| Detail page status bar | `solid` | Key metadata, glanceable |
| Table cell (single status) | `solid` | Clear status at a glance |
| Table cell (multiple tags) | `outline` | Reduce visual noise in dense rows |
| Card metadata | `outline` | Card border + solid badge = too heavy |

### Single selection among alternatives (compare flows)

When the user must **pick exactly one** option from a small set (e.g. compare two models, choose one deployment target), expose **one** primary selection affordance — not two competing ones.

**Preferred patterns (pick one):**

1. **Card selection only** — clicking the card sets its selected state through the framework-specific API and preserves an accessible native selection control.
2. **RadioGroup** — display options as a list of radio options using RadioGroup
3. **Dense compare** — **table row** selection or a single-column list with one control per row.

### Metric presentation

Rules for any displayed metric — stat blocks, table summary values, chart callouts, ranked-list columns. Composition for stat tiles lives in [stat-blocks.md](./stat-blocks.md) and for ranked columns in [ranked-lists.md](./ranked-lists.md); these formatting rules apply everywhere a number renders.

**The context triplet.** A headline metric renders **value + comparison + scope**. A bare number ("$4.82M" alone) forces the user to supply the baseline and the boundaries themselves — comparison is a delta vs. baseline, a share of a whole, or a distance to a target; scope is the period, denominator, or method that makes the value interpretable. In a table or ranked list the triplet may be distributed across columns (value column + share column + the panel's period subtitle) rather than packed into one element.

| Rule | Detail |
| --- | --- |
| Compact notation for large values | At ≥ 10,000 use compact one-decimal notation: `18.4K`, `4.8M`, `212.4B`. Keep full precision in tooltips and exports. |
| Currency precision | Cents only under ~$1,000 (`$13.76`); whole dollars to five digits (`$84,210`); compact above (`$4.82M`). |
| `tabular-nums` on aligned numbers | Any column of numbers, KPI values that update in place, and paired value/percent runs use `font-variant-numeric: tabular-nums` so digits align and don't jitter. |
| Right-align numeric columns | Tables and ranked lists right-align numbers so magnitudes compare down the column. |
| Name rate units explicitly | "$ / 1M tokens", "req / s" — never leave a rate's denominator implied. |
| Percentages | One decimal by default (`34.2%`); drop decimals only when everything on the surface is whole (`64%`). |
| No raw exponents or locale-less digits | `2.1e9` and `4820000` never render; format before display. |

### Automated insights list

A short list of computed signals ("Spend concentration," "Largest 7-day mover," "Anomaly detected") summarizing what changed in the selected range. This is **not** an [activity feed](./activity-feed.md) — an activity feed renders _history_ (events that happened); an insights list renders _derived analysis_ of the current range.

**Anatomy per insight:** a leading tone-appropriate icon (`aria-hidden`) + a bold one-line title + one sentence naming the **metric, magnitude, and timeframe** ("claude-fable-5 increased by $138.70 versus the prior 7 days"). One accent color plus a warning tone for anomalies is enough — don't invent a per-tone rainbow.


Icon selection and variant rules: [Icons](../ICONS.md).



**Rules:**

- Cap at ~4 insights per panel — beyond that it's a report, not a glanceable panel.
- Every insight is specific and quantified. "Usage changed significantly" is not an insight.
- When an insight is actionable, its row links to the filtered view that shows it.

---

## Modifiers

### Reorderable rows

If rows or items are reorderable by drag, provide a non-drag alternative — up/down buttons, a "move to…" action, or keyboard reordering — so ordering isn't drag-only. A drag interaction with no single-pointer alternative excludes anyone who can't perform a drag gesture.

### Progressive disclosure

Not all data needs to be visible at once. Pick the mechanism that matches how often users need the hidden content.

| Mechanism | Use when | Rules |
| --- | --- | --- |
| **Disclosure sections** | Content is detailed and rarely needed on initial view; users typically need only a subset; showing everything would overwhelm the primary content | Examples: environment variables, secrets, advanced configuration, a *secondary* audit log tucked below primary content. When activity/history is the primary surface, render it with [activity-feed.md](./activity-feed.md) instead of hiding it in a disclosure. |
| **Truncation with "Show more"** | Content is long but users sometimes need to see it all; full content would disrupt layout (e.g., long descriptions in cards) | Show enough to be useful (2–3 lines minimum); make truncation obvious (ellipsis + "Show more" link); expand in place, don't navigate away |
| **Separate tabs or pages** | Content is a complete alternate view (Logs, History, Metrics); users don't need both views simultaneously; each view has its own distinct purpose | Each view warrants its own route/tab |

### Empty and loading states

- For empty data displays: [empty-states.md](./empty-states.md)
- For loading skeletons / placeholders: [loading.md](./loading.md)
