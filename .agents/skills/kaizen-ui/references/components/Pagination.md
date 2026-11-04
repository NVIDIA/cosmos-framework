# Pagination

Navigation control that splits a long list of items across multiple pages. Pair with tables and grids that exceed a single page of content.

## Notes

- Only render Pagination when `totalItems > pageSize`. When the dataset fits on a single page, Pagination is unnecessary — conditionally render it.

- `totalItems` is required and drives the derived page count and item-range text. Pass it for every kind, including `kind="simple"`.

- Default `pageSize` to 10 for dense rows or 25 for regular rows; keep `pageSizeOptions` to `[10, 25, 50]` and add `100` only for power-user views.

- Pick `kind="input"` for large data tables where users may jump pages, `kind="tabs"` for a visible list of numbered page links or buttons, and `kind="simple"` for linear prev/next navigation through a short sequence.

- `kind="tabs"` is a visual variant and does not use tablist semantics; numbered pages render as links or buttons with `aria-current="page"`.

- The controls render inside a `navigation` landmark named "Pagination". Rename it via `attributes.PaginationNavigationGroup["aria-label"]`, giving each Pagination a distinct name when a page shows more than one so the landmarks stay distinguishable. The `kind="tabs"` page list is a `group` named "Pages".

- `kind="tabs"` never renders the item-range text even when `displayControls` is `true`; for `kind="input"` and `kind="simple"`, `displayControls` reveals the page-size select and item-range text.

- Quantitative thresholds: under 10 items skip pagination; 10–25 items consider a single page; above 25 items pagination is almost always appropriate.

## Examples

### Basic Pagination

Use for large datasets where users may need to jump to a specific page by number.

```tsx
<Pagination totalItems={100} defaultPage={1} defaultPageSize={10} />
```

### Tabs Pagination

Use for moderate-sized datasets where users benefit from seeing each available page.

```tsx
<Pagination kind="tabs" totalItems={100} defaultPage={1} defaultPageSize={10} />
```

### Simple Pagination

Use for linear navigation where only previous and next actions are needed.

```tsx
<Pagination kind="simple" totalItems={100} defaultPage={1} defaultPageSize={10} />
```

### With Controls Pagination

Use when users need to adjust page size and see the current item range.

```tsx
<Pagination displayControls totalItems={100} defaultPage={1} defaultPageSize={10} />
```

### Controlled Pagination

Pass `page` + `onPageChange` (and `pageSize` + `onPageSizeChange`) when pagination state lives outside the component — e.g. synced with URL params, a query cache, or a server response.

```tsx
<Pagination totalItems={100} page={1} pageSize={10} onPageChange={handlePageChange} onPageSizeChange={handlePageSizeChange} />
```

### With Custom Range Text Pagination

Pass `rangeTextFormatFn` to override the default `"1-10 of 100 items"` label. Receives `{ firstItemIndex, lastItemIndex, totalItems, pageSize }` and returns a `ReactNode`. Requires `displayControls`.

```tsx
<Pagination
  displayControls
  totalItems={100}
  defaultPage={1}
  defaultPageSize={10}
  rangeTextFormatFn={(range) => `Showing ${range.firstItemIndex}–${range.lastItemIndex} of ${range.totalItems} results`}
/>
```

### Tabs As Links Pagination

Use getHref to generate numbered-page and arrow destinations. Include a results-region fragment when navigation should return users to the updated content; use renderLink for a framework router link.

```tsx
<Pagination kind="tabs" totalItems={100} page={2} pageSize={10} getHref={({ page, pageSize }) => `?page=${page}&pageSize=${pageSize}#results`} />
```

### Form Submission Pagination

Use a shared GET form when changing pages should also submit current filter inputs. Keep a nameless submit button ahead of Pagination so Enter does not activate a page-size option or arrow, and resolve repeated pagination fields with `parsePaginationSearchParams` on the server.

```tsx
<form method="get" action="/results">
  <button type="submit" hidden />
  <input name="query" type="hidden" value="gpu" />
  <Pagination action="form" displayControls totalItems={100} defaultPage={1} defaultPageSize={10} />
</form>
```

### Cursor Based Pagination

Override `name` and `value` on individual arrow buttons via `attributes` when integrating with cursor-based APIs that submit opaque cursors instead of page numbers.

```tsx
<form>
  <Pagination
    kind="simple"
    action="form"
    totalItems={100}
    page={2}
    pageSize={10}
    attributes={{
      PaginationArrowButtonPrevious: {
        name: "cursor",
        value: "prev_abc123",
      },
      PaginationArrowButtonNext: {
        name: "cursor",
        value: "next_xyz789",
      },
    }}
  />
</form>
```

### Tabs Form Submission Pagination

Numbered pages submit their own page number under `action="form"`. Supply custom `items` rendered as `<Button type="submit" name="page">` when the pages need their own markup or styling.

```tsx
<form>
  <Pagination
    kind="tabs"
    action="form"
    totalItems={50}
    defaultPage={1}
    defaultPageSize={10}
    items={Array.from({ length: 5 }, (_, i) => ({
      value: i + 1,
      asChild: true,
      children: (
        <Button type="submit" name="page" value={i + 1} kind="tertiary">
          {i + 1}
        </Button>
      ),
    }))}
  />
</form>
```

### Linked Form Submission Pagination

Combine getHref for arrow and page-size links with a GET form for typed page entry. Page-size links reset to page 1. Preserve applied filters and page size in link URLs and hidden fields.

```tsx
<div className="flex flex-wrap items-center gap-4">
  <form action="/results" method="get" className="flex items-center gap-2">
    <input type="hidden" name="pageSize" value="10" />
    <input type="hidden" name="query" value="gpu" />
    <Pagination action="form" displayControls totalItems={100} page={2} pageSize={10} getHref={({ page, pageSize }) => `/results?page=${page}&pageSize=${pageSize}&query=gpu`} />
    <Button type="submit">Go</Button>
  </form>
</div>
```

### Composed

```tsx
<PaginationRoot totalItems={100} defaultPage={1} defaultPageSize={10}>
  <PaginationNavigationGroup>
    <PaginationArrowButton direction="first" />
    <PaginationArrowButton direction="previous" />
    <PaginationPageInput />
    <PaginationPageCountText
    // pageCountTextFormatFn={(pageMeta) => `of ${pageMeta.total}`}
    />
    <PaginationArrowButton direction="next" />
    <PaginationArrowButton direction="last" />
  </PaginationNavigationGroup>
</PaginationRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **displayControls** * | `boolean` | `false` | Shows the page-size select and the item-range text. |
| **totalItems** * | `number` | - | Total number of items in the underlying dataset. Used to derive the page count and item range. |
| action | `"form"` | - | Submits `page` and `pageSize` through the associated form instead of updating internal state. Use the pagination parsing helpers to resolve repeated fields from shared forms. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ PaginationRoot?: NativeElementAttributes<"div", typeof PaginationRoot>; PaginationNavigationGroup?: NativeElementAttributes<"div", typeof PaginationNavigationGroup>; PaginationControlsGroup?: NativeElementAttributes<"div", typeof PaginationControlsGroup>; PaginationArrowButtonFirst?: PaginationArrowAttributes; PaginationArrowButtonPrevious?: PaginationArrowAttributes; PaginationArrowButtonNext?: PaginationArrowAttributes; PaginationArrowButtonLast?: PaginationArrowAttributes; PaginationPageInput?: NativeElementAttributes<"input", typeof PaginationPageInput>; PaginationPageList?: NativeElementAttributes<"div", typeof PaginationPageList>; PaginationPageCountText?: NativeElementAttributes<"span", typeof PaginationPageCountText>; PaginationItemRangeText?: NativeElementAttributes<"div", typeof PaginationItemRangeText>; PaginationPageSizeSelect?: NativeElementAttributes<"button", typeof PaginationPageSizeSelect>; PaginationDivider?: NativeElementAttributes<"div", typeof PaginationDivider> }` | - | Native HTML attributes forwarded to the internal composed components. |
| defaultPage | `number` | `1` | Initial page when uncontrolled. |
| defaultPageSize | `number` | `10` | Initial page size when uncontrolled. |
| getHref | `(destination: { page: number; pageSize: number; }) => string` | - | Builds URLs for arrows, numbered pages, and page-size options. Page-size destinations use page 1. |
| items | `(Omit<TabItem, "value" \| "attributes"> & { value: number; attributes?: PaginationPageItemAttributes; })[]` | - | Overrides the list of page items, useful for rendering links or custom form submit buttons. Only applies when `kind` is `"tabs"`. |
| kind | `"input" \| "tabs" \| "simple"` | `"input"` | The style of pagination control. - "input" - First/previous/next/last arrows with a number input. Use for large datasets where users may jump to arbitrary pages. - "tabs" - First/previous/next/last arrows with a numbered page list. Use for moderate-sized datasets where users skim sequentially. - "simple" - Just previous and next buttons. Use for linear navigation through a sequence. |
| onPageChange | `(page: number) => void` | - | Called when the current page changes. |
| onPageSizeChange | `(pageSize: number) => void` | - | Called when the page size changes. |
| page | `number` | - | Controlled current page. When provided, the consumer is responsible for updating it in response to `onPageChange`. |
| pageMeta | `{ first: number; last: number; total: number }` | - | Overrides the computed page metadata. Use when the source of truth lives outside the component (e.g. cursor-based APIs). |
| pageSize | `number` | - | Controlled page size. When provided, the consumer is responsible for updating it in response to `onPageSizeChange`. |
| pageSizeOptions | `number[]` | `[10, 25, 50, 100]` | Selectable page sizes shown in the page-size select. |
| rangeMeta | `{ firstItemIndex: number; lastItemIndex: number; totalItems: number; pageSize: number }` | - | Overrides the computed item-range metadata. Use when the visible range cannot be derived from `page` and `pageSize`. |
| rangeTextFormatFn | `(rangeMeta: { firstItemIndex: number; lastItemIndex: number; totalItems: number; pageSize: number }) => ReactNode` | - | Formats the item-range text rendered alongside the controls. |
| renderedItemCount | `7 \| 5 \| 6 \| 8 \| 9 \| 10` | `7` | Total number of page controls to render, including ellipses. A fixed count prevents layout shift as the page changes. Only applies when `kind` is `"tabs"`. |
| renderLink | `(item: { href: string; children?: React.ReactNode; }) => ReactNode` | - | Renders navigation links with a custom component, typically a framework router link. |

`* = required prop`

## Exports

- Components: `Pagination`, `PaginationArrowButton`, `PaginationControlsGroup`, `PaginationDivider`, `PaginationItemRangeText`, `PaginationNavigationGroup`, `PaginationPageCountText`, `PaginationPageInput`, `PaginationPageList`, `PaginationPageListItem`, `PaginationPageSizeSelect`, `PaginationRoot`
- Utilities: `createPaginationSearchParams`, `getPageInputFieldName`, `parsePaginationFormData`, `parsePaginationSearchParams`
- Types: `CreatePaginationSearchParamsOptions`, `PageMeta`, `PaginationArrowButtonProps`, `PaginationControlsGroupProps`, `PaginationDividerProps`, `PaginationFormData`, `PaginationItemRangeTextProps`, `PaginationNavigationGroupProps`, `PaginationPageCountTextProps`, `PaginationPageInputProps`, `PaginationPageItem`, `PaginationPageListItemProps`, `PaginationPageListProps`, `PaginationPageSizeSelectProps`, `PaginationProps`, `PaginationRootProps`, `ParsePaginationFormDataOptions`, `ParsePaginationSearchParamsOptions`, `RangeMeta`, `SearchParamsInput`

## See also

- [Table](./Table.md)
