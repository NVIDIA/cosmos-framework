# Search pattern

A text input that lets users find content by query. Search often works alongside filters — read [`filters.md`](./filters.md) when both are present.

## When to use

Reach for search when items need to be found by name, keyword, or content rather than narrowed by attribute. See [`filters.md`](./filters.md) when attribute-based narrowing is the better tool, or when search and filters should coexist.

## Anatomy

A search input plus a results display. The non-obvious rule - **the query lives in the input only**. Do not duplicate it as a tag or chip elsewhere, unless it's part of an applied-filters display in a filter-level search context.

Optional parts, include when applicable:

| Part | Include when |
| --- | --- |
| **Inline validation message** | Field has character restrictions (e.g., no special characters allowed for certain query types). Be specific about which characters are disallowed |
| **Scope indicators** (tabs / chips / categories) | Content spans sub-types or content kinds. See [Modifiers → Scope indicators](#scope-indicators) |
| **Result count** | Useful when search updates content in place, especially combined with active filters, so the user sees the full query state |

## Variants

Pick the variant that matches the search scope. Variant choice drives placement and result behavior.

| Variant | Placement | Result behavior | Use when |
| --- | --- | --- | --- |
| **Global search** | Persistent search bar in the app header / AppBar; always accessible regardless of current page | Navigates the user to a dedicated results page. Group results by content type when the app has multiple types (e.g., "Models (94)", "Blueprints (34)", "Other (39)") | User searches across the entire application; user may not know which section has the target content; the app has multiple distinct content areas |
| **Contextual search** | Within or directly above the content area it searches; no filters present alongside | Content updates in place; no page navigation | User searches within a single data set or view on the current page; scope is limited to the current context |
| **Filter-level search** | Inside or adjacent to the filter controls; functions as part of the filter system | Content updates in place; query combines with active filters; both affect displayed results | Search is paired with filters; user needs to search by text and narrow by filter criteria simultaneously |

**Anti-pattern:** the same input handling global and contextual search depending on page state. Users can't predict scope from the input alone — use distinct placements (header for global, page-level for contextual).

## Query Behavior

- **Debounce** ~200–400ms; for high-latency datasets add a **2–3 character minimum** or require **Enter** to submit.
- **Rank results by relevance** (not alphabetical/unranked); offer a secondary sort if needed.
- **Zero results** → see [`empty-states.md`](./empty-states.md#no-results-filtered-or-searched): acknowledge the query, suggest next steps, and offer a reset rather than a bare "no results."

## Scope indicators

Provide scope indicators (tabs, chips, or categories) when content has sub-types or kinds. Show counts per category so users can narrow without re-querying.

| Variant | Scope indicator placement |
| --- | --- |
| **Global search** | Tabs / chips on the dedicated results page, grouped by content type with counts (e.g., Models, Blueprints, Other) |
| **Contextual / filter-level search** | Content-type tabs or toggles above the results when content spans sub-categories (e.g., a docs portal spanning Guides, Recipes, Reference, Changelog, Discussions) |
