# Table

Organizes sets of structured data into columns and rows so users can scan, compare, and analyze information.

## Notes

- When Table is the primary page content, render it directly inside the main content area without wrapping it in a Panel. Panel-wrapping is only appropriate for dashboard table widgets, or when table is a subsection of a page.

- For sorting, filtering, and other advanced data grid features, reach for a dedicated data grid component first if one is available in your library.

- Place the row's primary identifier (name, ID, email, file name) in the first data column. Status, priority, badge-only values, and decorative icons belong to the right of the identifier; selection checkboxes and expansion toggles may precede it.

- When a row has a single navigation action, make the entire row clickable via `onRowSelect` and enable `hoverableRows`. Do not add a redundant "View" column. For 2-3 actions per row, use a kebab Dropdown in the rightmost cell.

- Sort via clickable column headers (`onColumnSelect` on a column definition) — do not add both column-header sorting and a toolbar sort dropdown to the same table.

- Do not apply background colors at the table, row, or cell level in the resting state. The page background should show through; hover states still use a background change.

- To make a whole cell a link or button target, wrap the cell content in `TableCellBleed` (the cell's only direct child) with a single native `<a href>` or `<button>` inside it; the control's click area then covers the cell. Do not put `onClick` on the bleed or the cell, and do not use the bleed to add background colors.

- For ProgressBar inside a cell, use the small variant alongside the numeric value — the number is primary, the bar is glanceable supporting context. Never use a full-width or tall bar.

- Preferred defaults: `align="left"` for text and `align="right"` for numbers so decimals line up across rows. Avoid mixing alignments across columns when it would hurt readability.

- Enable `hoverableRows` only when rows are interactive (e.g. `onRowSelect` is set). On purely informational tables, hover is false affordance.

- Bulk-selection pattern: add a leading checkbox column and set TableToolbar.slotBulkActions for a contextual selection toolbar (count + bulk actions) while rows are selected using `showBulkActionsToolbar` on the TableToolbar. For 1 or 2-3 per-row actions, keep the existing single-action / kebab Dropdown patterns.

- In a row that pairs a title with a description, render the description in a smaller text-secondary style below the title so the eye lands on the title first. Limit at most one column per table to multi-line content — multiple multi-line columns reduce scanability and read as stacked cards.

- Responsive behavior: at lg+ render the full table; at md, allow horizontal scroll via `overflow-x-auto` with a `min-w-(--breakpoint-md)` table; at sm, consider an alternative layout such as a card list when the table no longer scans.

## Examples

### Basic Table

Use to display structured data in columns and rows for scanning and comparison. When helpful, you can pair the row's primary identifier with a smaller, secondary-colored description in the first data column. Reserve this title-plus-description pattern for the identifier column only

```tsx
<Table
  columns={["Name", "Role", "Status"]}
  rows={[
    {
      id: "1",
      cells: [
        {
          children: (
            <Flex direction="col">
              <Text kind="body/bold/md">Alice</Text>
              <Text kind="body/regular/sm" className="text-secondary">
                Joined 2026-01-01
              </Text>
            </Flex>
          ),
        },
        "Engineer",
        {
          children: (
            <Badge color="green" kind="solid">
              Active
            </Badge>
          ),
        },
      ],
    },
    {
      id: "2",
      cells: [
        {
          children: (
            <Flex direction="col">
              <Text kind="body/bold/md">Bob</Text>
              <Text kind="body/regular/sm" className="text-secondary">
                Joined 2026-01-01
              </Text>
            </Flex>
          ),
        },
        "Designer",
        {
          children: (
            <Badge color="green" kind="solid">
              Active
            </Badge>
          ),
        },
      ],
    },
  ]}
/>
```

### Hoverable Rows Table

Use when rows are interactive or clickable and a hover state helps indicate interactivity. Combine with `selected` to highlight the selected row for example - useful when users need to select rows for bulk actions or detail views.

```tsx
() => {
  const [selectedRow, setSelectedRow] = useState("1");
  return (
    <Table
      hoverableRows
      columns={["Name", "Role", "Status"]}
      rows={[
        { id: "1", cells: ["Alice", "Engineer", "Active"] },
        { id: "2", cells: ["Bob", "Designer", "Active"] },
        { id: "3", cells: ["Charlie", "Manager", "Inactive"] },
      ].map((row) => ({
        ...row,
        onRowSelect: () => setSelectedRow(row.id),
        selected: selectedRow === row.id,
      }))}
    />
  );
};
```

### Auto Layout Table

Use when column widths should adapt to their content rather than being evenly distributed.

```tsx
<Table
  layout="auto"
  columns={["ID", "Description", "Updated"]}
  rows={[
    { id: "1", cells: ["NV-001", "GPU cluster provisioning", "2 hours ago"] },
    {
      id: "2",
      cells: ["NV-002", "Model training pipeline", "5 minutes ago"],
    },
  ]}
/>
```

### Table Density

Configure table density to be appropriate for the context. This can reduce readability and should only be used sparingly.

```tsx
<Table
  columns={["Metric", "Value", "Change"]}
  // Set density to compact when vertical space is at a premium, or spacious when there is a lot of vertical space.
  density="compact"
  // Cells default to a minimum content height; set cellHeight="dense" to have cells fit their content instead.
  cellHeight="dense"
  rows={[{ id: "1", cells: ["Throughput", "1,200 req/s", "+12%"] }]}
/>
```

### With Toolbar Table And Bulk Actions

Pair a Table with a TableToolbar to host primary actions and a bulk-action bar. Toggle showBulkActionsToolbar based on whether any rows are selected to swap between the two states for bulk actions.

```tsx
<Flex direction="col" gap="density-md">
  <TableToolbar
    showBulkActionsToolbar={HAS_SELECTED_ROWS}
    slotBulkActions={
      <>
        <Text>1 user selected</Text>
        <Button color="danger" size="small">
          Delete
        </Button>
      </>
    }
  >
    <Text>1 user</Text>
    <Button color="brand">Add New</Button>
  </TableToolbar>
  <Table columns={["Name", "Email"]} rows={[{ id: "1", cells: ["Alex Morgan", "alex@example.com"] }]} />
</Flex>
```

### Table With Cell Bleed

Wrap a link in TableCellBleed to make the whole cell its click target. The link text keeps its place in the column, and the bleed draws an inset focus ring. A lone `<a href>` or `<button>` inside the bleed stretches automatically; set `padding="none"` when the child fills the cell itself, like a `Button` sized to the cell.

```tsx
<Table
  columns={["Instance", "GPU", "Status"]}
  rows={[
    {
      id: "1",
      cells: [
        {
          children: (
            <TableCellBleed>
              <a href="#training-a100-01">training-a100-01</a>
            </TableCellBleed>
          ),
        },
        "NVIDIA A100",
        "Running",
      ],
    },
    {
      id: "2",
      cells: [
        {
          children: (
            <TableCellBleed>
              <a href="#training-h100-02">training-h100-02</a>
            </TableCellBleed>
          ),
        },
        "NVIDIA H100",
        "Stopped",
      ],
    },
  ]}
/>
```

### Composed

```tsx
<TableRoot hoverableRows>
  <TableHead>
    <TableRow>
      <TableHeaderCell>Name</TableHeaderCell>
      <TableHeaderCell>Role</TableHeaderCell>
      <TableHeaderCell>Status</TableHeaderCell>
    </TableRow>
  </TableHead>
  <TableBody>
    <TableRow>
      <TableDataCell>Alice</TableDataCell>
      <TableDataCell>Engineer</TableDataCell>
      <TableDataCell>Active</TableDataCell>
    </TableRow>
    <TableRow>
      <TableDataCell>Bob</TableDataCell>
      <TableDataCell>Designer</TableDataCell>
      <TableDataCell>Active</TableDataCell>
    </TableRow>
  </TableBody>
</TableRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **columns** * | `(string \| ({ children: ReactNode; onColumnSelect?: (_: { columnIndex: number; }) => void }))[]` | - | Column headers rendered in order. Pass a string for plain text or an object to attach selection callbacks and attributes. |
| **rows** * | `{ id?: string; cells: (string \| ({ children: ReactNode; onCellSelect?: (_: { rowId?: string; columnIndex: number; }) => void }))[]; onRowSelect?: (_: { rowId?: string; }) => void; selected?: boolean }[]` | - | Row data rendered in order beneath the headers. |
| align | `"center" \| "right" \| "left"` | `"left"` | Horizontal alignment applied to every cell in the table. |
| attributes | `{ TableHead?: NativeElementAttributes<"thead", typeof TableHead>; TableRow?: NativeElementAttributes<"tr", typeof TableRow>; TableBody?: NativeElementAttributes<"tbody", typeof TableBody> }` | - | Native HTML attributes forwarded to the internal composed components. |
| cellHeight | `"default" \| "dense"` | `"default"` | Controls the minimum height of cell content. - "default" - Cells reserve a minimum content height so rows stay a consistent height. - "dense" - Cells shrink to fit their content, for compact data-dense tables. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces. |
| hoverableRows | `boolean` | `false` | Applies a hover style to body rows on pointer-over. |
| layout | `"auto" \| "fixed"` | `"fixed"` | Controls the CSS `table-layout` algorithm. |
| ref | `Ref<HTMLTableElement>` | - |  |

`* = required prop`

## Exports

- Components: `Table`, `TableBody`, `TableBulkActionToolbar`, `TableCellBleed`, `TableDataCell`, `TableHead`, `TableHeaderCell`, `TableRoot`, `TableRow`, `TableToolbar`
- Types: `TableBodyProps`, `TableBulkActionToolbarProps`, `TableCellBleedProps`, `TableCellData`, `TableColumnDefinition`, `TableDataCellProps`, `TableHeaderCellProps`, `TableHeadProps`, `TableProps`, `TableRootProps`, `TableRowDefinition`, `TableRowProps`, `TableToolbarProps`

## See also

- [Pagination](./Pagination.md)
