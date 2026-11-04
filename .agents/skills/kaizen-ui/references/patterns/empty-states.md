# Empty States

Empty states aren't just "nothing here" messages — they orient the user, explain the gap, and route them to a next action. A good empty state reduces confusion and prevents dead ends.

## When to use

Use an empty state when:

- A content area (page, section, table, chart, list) has no items to render
- The reason is reachable in normal usage — first-use, filtered/searched, no permission, no data in scope, or a recoverable error
- The user needs guidance, recovery, or context to understand why nothing is shown

Don't use when:

- Data is still loading — use [ProgressBar](../components/ProgressBar.md) or [Skeleton](../components/Skeleton.md) instead
- A hard failure would replace a full page — use the [error page layout](../layouts/error-page.md), not this pattern
- Container-level load/retry errors aren't yet scoped — see [`error-states.md`](./error-states.md); that pattern owns the scope decision (full-page vs. container vs. banner vs. inline). Use the [Something Went Wrong](#something-went-wrong-error) variant here only after `error-states.md` has resolved that the right scope is **container-level**.

---

## Anatomy

Every empty state shares the same core slots. Variants differ in content values, not in which slots are present. In most cases the [StatusMessage](../components/StatusMessage.md) component should be used to render these slots.

### Required

| Part | Purpose |
| --- | --- |
| **Heading** | States what's empty or what this area is for ("No locations yet", "No results for 'xyz'"). |
| **Subheading** | 1–2 sentences explaining context, purpose, or recovery. |

### Optional

Include when the variant warrants them.

| Part | Include when |
| --- | --- |
| **Primary action** | A clear next step exists (create first item, clear filters, retry, request access). Rendered in the footer/action region. |
| **Secondary action** | A genuinely useful alternative path (docs link, contact support) — see [Composition rules → Action hierarchy](#action-hierarchy). Rendered in the footer/action region. |
| **Illustration** | First-use moments where orientation/encouragement helps. Simple, on-brand, small — never dominant. Rendered in the media region. |
| **Icon** | Lighter alternative to an illustration; appropriate for inline / component-level empties. Rendered in the media region. |
| **Active filter visibility** | A "No Results" variant — keep applied filters/search terms visible above the empty state so the user sees what's restricting results - these should be rendered outside the StatusMessage, near the filter controls themselves. |

---

## Variants

Identify which variant applies based on **why** the content area is empty.

### First-Use / No entries yet

When the area has no items - totalCount is 0 because none exist, not because of error or applied filters. In this case provide the user with onboarding guidance. Do not render empty tables or useless filters/inputs.

The heading should speak to the collection(ex. "No clusters yet"), subheading should be 1-2 sentences answering "why would I add something here?"(ex. "Add a cluster to start running workloads, monitor usage, and manage capacity across your team."), and a prominent primary CTA for users to begin.

**Caveat**: Sometimes a first-use/onboarding step doesn't make as much sense. For example, a monitoring chart with no recent activity. In that case the subheading is less "why would I add something here" and "when would something appear here" - ex. "No activity in the last 24 hours."

### No Results (filtered or searched)

When the area has no items because active filters/search filtered results down to 0. In this case do not render an empty table, but keep rendering any filter controls.

The heading should name the miss (e.g., "No results for 'xyz'"), the subheading should suggest broadening terms or removing a filter, and there should be a single secondary [Button](../components/Button.md) labeled "Clear filters" or "Clear search". Keep applied filters/search visible so the user sees what's restricting results.

### Something Went Wrong (error)

For "something went wrong" empty states (load failures, retry-able errors), see [`error-states.md`](./error-states.md) → Container-level errors. Resolve the scope decision (full-page vs. container vs. banner vs. inline) there first; if container-level is correct, render the empty state using this variant.

### No Access / Unauthorized

When the content exists but is restricted to roles/permissions the user lacks.

The heading should state what's happening (for example, "You don't have access to this section"), the subheading should explain what permission is needed or who to contact, and, if possible, provide a "Request access" primary action when a self-service flow exists. Otherwise use an [Anchor](../components/Anchor.md) to the right admin or support channel. Use a lock icon in the media region. Hide create actions the user cannot access.

### Gotchas

- Do not show a generic "No data" graphic for both empty collections and filtered to empty, both states have different recovery actions.
- Do not render "0 items" in the result count while loading - hide it until data is loaded
- Do not use two brand-colored buttons in one StatusMessage. They compete for attention; keep one primary action per state.

---

## Composition rules

Apply to all variants.

### Action hierarchy

When an empty state includes actions, follow this priority:

1. **Primary action** — the one thing the user most likely wants to do (create the first item, clear filters, retry loading). Use [Button](../components/Button.md) with the brand color — NVIDIA green is the CTA signal.
2. **Secondary action** — an alternative next step (view documentation, contact support, adjust settings). Show as an [Anchor](../components/Anchor.md) or secondary Button.
3. **Never more than 2 actions** — an empty state with 3+ buttons creates decision paralysis at a moment when the user is already uncertain.

### Same anatomy across variants

Every empty state in the same surface uses the same core slots (headline + supporting text, with optional action + visual). Variation comes from content **values** per variant, not from which slots are present. Don't drop the headline or supporting text to "simplify" an error or no-results state — it's the slot consistency that makes the pattern legible.

---

## Modifiers

### Placement

| Placement | Use when |
| --- | --- |
| **Full-container** | The entire content area of a page or section is empty. Center the headline / text / CTA both horizontally and vertically within the container the content would normally occupy. |
| **Inline** | A specific component within a page is empty but the rest of the page has content (a chart with no data while other charts do, a "No comments yet" section on a detail page). Smaller and simpler — typically just a short message and optional action, without illustrations. |
| **Table** | A table has no rows to display. Show the table header (column names) so the user understands the structure; render the empty state message centered in the table body area; keep the filter/search controls above the table so the user can adjust their query. |

### Visual treatment

| Treatment | Use when |
| --- | --- |
| **Illustration** | First-use empty states where the user needs encouragement or orientation. Keep simple, on-brand, small — never dominating the empty state. Do **not** use for error or no-results states — those are functional moments, not onboarding moments. |
| **Icon** | Lighter alternative to an illustration (search icon for no results, warning icon for errors). Appropriate for inline empty states and component-level empties. |
| **No visual** | Dense/operational interfaces where empty states are frequent and expected (filtered table views in admin tools). Just the headline + supporting text + action. |
