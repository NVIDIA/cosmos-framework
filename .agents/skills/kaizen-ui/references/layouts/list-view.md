# List View (Layout Reference)

A filterable, sortable table with search, optional scope tabs, and pagination. This is the most common browse pattern in NVIDIA enterprise applications.

For inline / region / page-level error scope decisions see [error-states.md](../patterns/error-states.md).

## When to use

- URL represents a collection (e.g., `/clusters`, `/functions`, `/models`)
- Users filter, search, and sort to find specific items
- Each row represents a single entity the user can act on or navigate to

**Examples:** GPU cluster inventory, function versions, deployment list, user directory, job history, API key management.

## Canonical Layout

1. Start with the [stacked shell](../layouts/page-shell.md#stacked-shell).
2. Add a [PageHeader](../components/PageHeader.md) with the collection title, optional description, and collection-level actions such as Create or Export. Row-level actions such as Configure or Edit stay with the relevant row.
3. Add scope tabs only for mutually exclusive collection subsets. If subsets overlap or ordinary filtering expresses the distinction, omit tabs.
4. Add a toolbar with [search](../patterns/search.md) followed by up to three common [filters](../patterns/filters.md). Use 320px as the search-width default and 176px for compact filters.
5. Add a result-count [Text](../components/Text.md) with `aria-live="polite"`. Format it as `{count} {collection name}` and include total context when filtered, for example "12 of 150 functions."
6. Render [Table](../components/Table.md) as the primary content according to [data display](../patterns/data-display.md). Native links and buttons inside rows own navigation and actions.
7. Sort a primary table through column headers. Use a separate sort control only for card grids or simple lists; never expose both mechanisms for the same data.
8. Add [Pagination](../components/Pagination.md) only when results exceed one page. Expose the item range and, when useful, items per page.

## Accessibility

- **Search and filter labels:** every [TextInput](../components/TextInput.md) and [Select](../components/Select.md) needs an explicit accessible name. Placeholders are not labels.
- **Result count live region:** put `aria-live="polite"` on the result-count text or its container so screen readers announce query changes.
- **Focus after clearing filters:** return focus to the search input or the first filter control, not the result count or the table.

## Responsive behavior

For toolbar collapse rules, see [filters.md → Horizontal bar layout contract](../patterns/filters.md#horizontal-bar-layout-contract). Let tables scroll horizontally when columns cannot reflow, and simplify pagination on narrow viewports.

## Alternative States

A list view is one of these states at a time. The matrix below decides what to render in each; it resolves the "empty table vs. message" question — only the **filtered-empty** state keeps the table structure.

| State | Toolbar (search + filters) | Table | Primary content | Reference |
| --- | --- | --- | --- | --- |
| **Loading** | Skeleton matching control sizes | Skeleton rows matching columns | — (omit result count; never `0 items`) | [Loading](#loading) |
| **First-use** (totalCount 0, none exist) | Hidden — no data to filter | Hidden — do **not** render an empty table | `StatusMessage` with onboarding + primary CTA | [zero state](../patterns/empty-states.md#first-use-no-entries-yet) |
| **Filtered-empty** (filters/search → 0) | **Visible** — keep filters + applied-filter chips so the user sees what's restricting | Header may stay; rows replaced | `StatusMessage` "No results" + "Clear filters" | [filtered empty state](../patterns/empty-states.md#no-results-filtered-or-searched) |
| **No access** | Hidden | Hidden | `StatusMessage` with permission guidance; hide the primary page action | [permission denied state](../patterns/empty-states.md#no-access-unauthorized) |
| **Load error** | Hidden (or keep if partial) | Hidden | `StatusMessage` + Retry (scope per error-states) | [load error state](../patterns/empty-states.md#something-went-wrong-error) |
| **Populated** | Visible | Visible | Table rows + result count + pagination | [data display](../patterns/data-display.md) |

### Loading

- Skeleton heights and widths must match the rendered control sizes. A toolbar skeleton wider than the toolbar will shift the layout when data arrives.
- Skip the page-heading skeleton because its content is static. Render skeletons only for dynamic content.
- Omit the result count until data loads. Never render `0 items` while loading.

## React example

```tsx
<Stack gap="4">
  <PageHeader
    className="mb-2"
    kind="flat"
    slotHeading="GPU clusters"
    slotDescription="Browse and manage fleet capacity."
    slotActions={
      <Button color="brand">
        <Icon name="add" />
        Add
      </Button>
    }
  />

  {/* Filter toolbar (always shown here). Hide it only in a true zero state
	(no clusters at all), but keep it visible when active filters produce an empty result. */}
  <Flex gap="2" align="center" wrap="wrap">
    <TextInput aria-label="Search clusters" className="w-80" placeholder="Search by name" slotStart={<Icon name="magnifying-glass" />} type="search" />
    <Select aria-label="Filter by status" attributes={{ SelectTrigger: { className: "w-44" } }} dismissible items={statusItems} placeholder="Status" />
    <Select aria-label="Filter by team" attributes={{ SelectTrigger: { className: "w-44" } }} dismissible items={teamItems} placeholder="Team" />
    <Select aria-label="Filter by GPU type" attributes={{ SelectTrigger: { className: "w-44" } }} dismissible items={gpuItems} placeholder="GPU type" />
    {hasAppliedFilters && (
      <Button kind="tertiary" size="small">
        Clear Filters
      </Button>
    )}
  </Flex>

  <TableToolbar>
    <Text kind="title/sm" aria-live="polite">
      {clusterRows.length} clusters
    </Text>
  </TableToolbar>
  <div className="w-full overflow-x-auto">
    <Table className="w-full" columns={["Name", "Status", "Team", "GPU type", "Allocated"]} hoverableRows layout="auto" rows={clusterRows} />
  </div>
</Stack>
```
