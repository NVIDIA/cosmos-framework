# Browse Page (Layout Reference)

Use for pages where users find, filter, and select from a collection of items.

This file is a **router**: it helps you pick how results display, then defers to the file that owns each variant's markup. There is no standalone browse code example — every variant is built from the list-view shell plus a result-display pattern documented elsewhere.

---

## When to use

- URL represents a collection (e.g., `/tasks`, `/models`, `/functions`)
- Primary user intent is finding or comparing items
- Page shows multiple items of the same type
- User may filter, sort, or search to narrow results

**Examples:** task list, model catalog, function registry, user directory, order history.

---

## Choose the right variant

Browse pages share a common structure (page shell, header, filters, results) but differ in how results are displayed and how selection works.

See [data-display.md](../patterns/data-display.md) for the Table vs. Cards decision.

| Variant | When to use | Reference |
|---|---|---|
| **List view** | Results are a **table**. Users filter, sort, and act on rows. Most common browse pattern in NVIDIA enterprise apps. | [list-view.md](./list-view.md) |
| **Card grid** | Results are **cards** with visual content (images, icons, charts). Users browse to discover or compare. | See [Card grid variant](#card-grid-variant) below |
| **List → detail** | Selecting an item opens a contextual side panel instead of navigating away. Users compare items or take quick actions without losing list context. | See [List → detail variant](#list--detail-variant) below |
| **Hierarchical** | Items have **parent-child relationships** that expand inline. | See [Hierarchical variant](#hierarchical-variant) below |

**Default:** if unsure, start with **list view**. Tables are the standard for data-dense enterprise interfaces.

## Common structure (all variants)

All browse variants share the stacked page shell from [layout-principles.md](./layout-principles.md).

## Card grid variant

Same shell and toolbar as [list-view.md](./list-view.md) (page header, search + filters, result count) — the only change is that results render as a card grid instead of a `Table`. Build the cards from [entity-cards.md](../patterns/entity-cards.md) + [Card.md](../components/Card.md). No separate code example is needed.

Use when items have visual elements (images, thumbnails, charts) and users browse to discover or explore rather than searching for a specific item.

### Card grid rules

For card grid layout (columns, vertical stack) see [entity-cards.md](../patterns/entity-cards.md#modifiers); for shared anatomy and equal heights across the grid see [entity-cards.md](../patterns/entity-cards.md#composition-rules); for clickable-card vs. inline-actions rules see [Card.md](../components/Card.md).


Browse-page-specific: use `<Grid colMinWidth={280}>` for a **280px minimum card width** and automatic responsive columns.







### Filter placement

For filter count thresholds and toolbar collapse rules, see [filters.md](../patterns/filters.md).

## List → detail variant

This is the [list-view.md](./list-view.md) layout (table, toolbar, optional scope tabs) plus a [SidePanel](../components/SidePanel.md) — selection opens the panel instead of navigating away. The panel is **toggleable, not always-visible**: the list is primary content and the panel appears in response to selection.

For the IA decision (panel vs. standalone page vs. both, what content belongs in each), see [detail-placement.md](./detail-placement.md).


For `SidePanel` mechanics such as modal behavior, `closeOnClickOutside`, initial selection, headings, and full-detail escalation, see [SidePanel](../components/SidePanel.md).







## Hierarchical variant

Items expand inline to show child items. Use when:

- Data has parent-child relationships
- Users need to see children in the context of siblings
- Comparison across hierarchy levels matters

Rules:

- One level of inline expansion max. Deeper hierarchy → navigate to a dedicated view.
- Nested filtering only when children are numerous (>10) with distinct filter dimensions.

## Empty states

All browse variants share the same empty state patterns. See [list-view.md](./list-view.md) → variants and [empty-states.md](../patterns/empty-states.md).
