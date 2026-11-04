# Filters & Sort

Filters narrow a dataset to the user's relevant subset; sort orders what remains. This file covers when filters are warranted, their anatomy, placement variants, and how applied state, URL persistence, and sort compose with the result set.

## When to use

Use filters when results require narrowing by attribute before users can find what they want — datasets too large to scan, multiple dimensions users care about, or shareable filtered views that benefit from URL state.

Don't use when:

- Results fit on one screen and users typically want all of them
- Only one attribute matters — use search instead
- The dataset is small enough that scanning beats filtering

---

## Anatomy

A filter surface is composed of required parts plus optional parts included when applicable. Variation comes from [placement](#variants), not from which parts are present.

### Required

| Part | Purpose |
| --- | --- |
| **Filter controls** | One control per filter dimension. Control type depends on the dimension's shape — see [Filter control types](#filter-control-types). |
| **Result count** | Live count reflecting the filtered dataset (e.g., "Displaying 1,147 results", "200 Users"). Always visible. |
| **Applied-filter chips** | Tags/chips showing currently active filters, rendered below or adjacent to the controls whenever 1+ filter is active. Each chip shows the filter **category and value** (or **category and count** if multi-select) and includes a **close/remove** affordance to deselect that filter. |
| **Clear-all action** | Surface inline above the result set whenever 2+ filters are active. Do not bury inside a menu. |

### Optional

| Part | Include when |
| --- | --- |
| **Sort control** | Content has at least one parameter it can be meaningfully ordered by (date, name, popularity, relevance, size). See [Modifiers → Sort](#sort). |
| **Typeahead within a filter** | A filter category contains more than 7 options — lets users find a value without scrolling through a long list. |

### Filter control types

For each filter dimension, pick the control that matches the dimension's shape. **Pick one control type per dimension** — never mix (e.g., some chips and some checkboxes for the same set).

| Control | Use when |
| --- | --- |
| **[Checkbox](../components/Checkbox.md)** | Multi-select within a single category |
| **[Radio](../components/RadioGroup.md)** | Single-select within a category AND layout is **vertical sidebar** (options stacked and fully visible) |
| **[Select / Dropdown](../components/Select.md)** | Single-select within a category when layout is **horizontal**, or when vertical space needs to be conserved |
| **[Tag / Chip](../components/Tag.md)** | Surfacing a category + selected value/count ("Type: Model", "Type: 2 selected"), or rendering applied filters with remove affordances |
| **[Slider](../components/Slider.md)** | Filter value falls within a fixed numeric range (price, date range, learning rate, batch size) |
| **[Toggle / Switch](../components/Switch.md)** | On/off states ("Download Available: on/off", "API Endpoint: on/off") |
| **[Combobox](../components/Combobox.md)** | A filter category has more than 7 options |

**Anti-pattern:** Mixing control types within a single filter dimension (some chips, some checkboxes). Inconsistent affordance suggests filters behave differently when they don't.

---

## Variants

Placement variants. Choose by filter count, content type, and user intent.

| Variant | Use when |
| --- | --- |
| **Facet Panel** | More than 7 filterable dimensions, AND results are cards / browsable collections, AND filters affect the whole page globally, AND user intent is exploration / discovery |
| **Horizontal bar** | 7 or fewer dimensions, OR results are tables / specific item lists, OR user intent is targeted search |
| **Inline / component-level** | Page contains multiple independent data views (e.g., a dashboard with charts, tables, and graphs over different data sources) AND each view has its own data structure AND a single global filter set cannot logically apply |

**Default:** When in doubt, default to **horizontal bar**. This aligns with the predominant pattern across NVIDIA enterprise products and is the safest choice for consistency.

Keep active filters visible in both variants — applied-filter tags and Clear filters belong above the result set, not hidden in the control that opened them.

### Facet Panel

Facet Panels are more scalable — they support nested values inside expandable sections by being built on `Accordion`. They keep all filter options visible and scannable, supporting progressive disclosure through expand/collapse. These should be rendered alongside the primary content as a sidebar, and on mobile collapse into a drawer with a prominent filter trigger.

The mobile filter trigger should look like a `Filter by` rail with an inline reset action.

Layout:

- Page shell uses the 80rem (1280px) browse-page maximum.
- Filter categories are accordion sections with stacked controls.
- Results column starts with title/count plus Sort aligned right.

> **KUI convention note:** The threshold of 7 is a KUI convention from a typical 80rem (1280px) page shell and 176px controls; the real constraint is available horizontal space, so adjust if labels or controls are unusually narrow or wide.

### Horizontal bar

Horizontal bars preserve content width and keep the user's attention focused on results. They are the default for most NVIDIA enterprise product interfaces. They work well on mobile but may lack context on available options until opened.

See [Composition rules → Horizontal bar layout contract](#horizontal-bar-layout-contract) for the responsive layout requirements.

Layout:

- Page shell remains at the 80rem (1280px) browse-page maximum.
- Use a row of up to three `Select` controls for common dimensions.
- Render applied-filter `Tag`s with close affordances below/near the controls as soon as filters are active.
- Show `Clear filters` inline beside applied tags.

**Anti-pattern:** Inline filter bar with 8+ controls — items wrap or push past the viewport and row breaks become visual noise. Switch to vertical sidebar at 8+ filter dimensions.

### Inline / component-level

Inline filters are contextualized to the specific component they control. Use only when the page has multiple independent data views whose data structures differ enough that no global filter set applies.

---

## Composition rules

Apply to all variants.

### Apply real-time by default

Apply filters immediately as the user selects them; results update in real time. Use an explicit **Apply Filters** button only when:

- The dataset is large enough that each filter change triggers a costly server call, OR
- Applying filters in real time would cause noticeable performance degradation (perceptible delay greater than ~300ms per filter change)

When using explicit apply, also include a **Reset** / **Clear Filters** action.

**Anti-pattern:** Apply button on simple toggles or single-select dropdowns — forces the user to commit a click that adds no value and slows iteration. Reserve Apply patterns for queries too expensive to re-run on every change.

### Reflect filter state in the URL

Filter state must be reflected in URL params so users can share a filtered view and don't lose state on refresh.

### Synchronize filter chrome with input state

Active-filter labels and counts (e.g., "Active filters: 3") must always match the actual input state.

### Empty / error states

When filters or search return zero results, apply the rules in [`empty-states.md`](./empty-states.md).

### Coexisting with search

When a search input sits alongside filters, treat them as independent controls over the same result set. See [`search.md`](./search.md) for search-specific anatomy and variants.

- Both the active query and active filters stay visible; **result count reflects the combined effect** of search + filters.
- Clearing search does **not** clear filters, and clearing filters does **not** clear the search.
- Place search in the same control group as the primary filters, **search first**, then filter controls. In the Figma list-view frame: search, followed by three `Select` controls. See [`../layouts/list-view.md`](../layouts/list-view.md) → Toolbar.

### Horizontal bar layout contract

See [`../layouts/list-view.md`](../layouts/list-view.md) → Toolbar section for structure.

### Keep filters discoverable

Show filters by default; hide behind a toggle only when they genuinely don't fit.

---

## Modifiers

### Sort

Sort is a separable affordance from filters. Include it when the content set has at least one parameter it can be meaningfully ordered by (date, name, popularity, relevance, size).

**Placement:** Sort appears as a separate control from filters — typically a dropdown aligned to the right side of the filter bar or results header. **It should not be mixed into the filter controls.**

#### Sort defaults

| Content shape | Default sort |
| --- | --- |
| Time-based (activity logs, recent items, changelogs) | **Newest first** |
| Search-result based (search query active) | **Relevance** |
| Browsable catalog, no search query active | **Most popular** / **Recommended** if popularity/recommendation data exists; otherwise **Alphabetical (A–Z)** |

#### Sort options

Offer sort options that map to real content attributes (Newest/Oldest, A–Z/Z–A, Most Popular, Size/Count, etc.). Two rules: offer **Relevance only when a search query is active**, and never offer a sort with no meaningful data behind it.
