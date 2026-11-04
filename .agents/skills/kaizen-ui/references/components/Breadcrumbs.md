# Breadcrumbs

A navigation trail that shows the user's location within the app hierarchy and links back to parent pages. Use concise labels for each level.

## Notes

- Do not render Breadcrumbs on root or first-level pages — there is nothing to navigate back to.

- All parent entries must be clickable links so users can navigate back; only the final entry (current page) should be a plain label.

- Keep labels concise — abbreviate or shorten long page titles so the trail fits on one line.

- Include Breadcrumbs on any page deeper than one level, and place them in PageHeader's `slotBreadcrumbs`.

## Examples

### Basic Breadcrumbs

```tsx
<Breadcrumbs items={[{ children: <a href="/">Home</a> }, { children: <a href="/category">Category</a> }, "Current Page"]} />
```

### With Icon

Icons should be used sparingly, but the component supports them.

```tsx
<Breadcrumbs
  items={[
    {
      children: (
        <a className="flex gap-[inherit]" href="/">
          <Home />
          Home
        </a>
      ),
    },
    { children: <a href="/category">Category</a> },
    {
      children: (
        <>
          <Identification />
          Product 123
        </>
      ),
    },
  ]}
/>
```

### Composed

```tsx
<BreadcrumbsRoot>
  <BreadcrumbsItem>
    <a href="/">Home</a>
  </BreadcrumbsItem>
  <BreadcrumbsSeparator />
  <BreadcrumbsItem>
    <a href="/category">Category</a>
  </BreadcrumbsItem>
  <BreadcrumbsSeparator />
  <BreadcrumbsItem active>Current Page</BreadcrumbsItem>
</BreadcrumbsRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `(string \| { children?: ReactNode })[]` | - | The breadcrumb entries, ordered from the root page to the current page. Strings render as plain text labels; objects allow custom content and per-slot native attributes. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Font size of the breadcrumb items. - "small" - 12px, for dense layouts. - "medium" - 14px, the standard size. - "large" - 16px, for prominent navigation. |
| slotSeparator | `ReactNode` | - | Replaces the default chevron separator between items with custom rendered content. |

`* = required prop`

## Exports

- Components: `Breadcrumbs`, `BreadcrumbsItem`, `BreadcrumbsRoot`, `BreadcrumbsSeparator`
- Types: `BreadcrumbsItemFieldProps`, `BreadcrumbsProps`, `BreadcrumbsRootProps`

## See also

- [PageHeader](./PageHeader.md)
