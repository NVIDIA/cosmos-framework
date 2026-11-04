# KeyValueGroup

A description list of labeled values, such as the properties of one resource. Renders a native `<dl>`.

## Notes

- Reach for KeyValueGroup to show the properties of a single entity (a project's owner, region, status). Use Table when readers compare the same fields across many entities, and List for items that are not label-value pairs.

- The default is a stacked group whose keys line up in one column. Set `pairOrientation="vertical"` for a narrow column such as a sidebar, and `orientation="horizontal"` for a compact row of stats.

- `id` is optional when `slotLabel` is a string; it defaults to the label. Pass `id` when the label is a node or when two pairs share a label.

- `slotValue` is any node. Pass a `Badge` for status, a `Tag` with `readOnly` for a category, or an `Avatar` before the name for a person. Mark that `Avatar` `aria-hidden`, since the name beside it already says who it is. Do not wrap plain text in `Text`; the value already carries its typography.

- Give the group a visible heading. A `<dl>` has no name of its own, and screen readers expose `aria-label` on it inconsistently.

- Pass `readOnly` to a `Tag` used as a value. A default `Tag` renders a `<button>`, and a focusable button with no action fails accessibility.

- Render an em dash with visually hidden "Not provided" text for a missing value rather than leaving the `<dd>` empty. Status in a value is a `Badge` with text, never color alone.

- When composing, render only `KeyValueGroupItem` directly inside `KeyValueGroupRoot`, and inside each item one `KeyValueGroupKey` followed by its `KeyValueGroupValue`. A `<dl>` holds either bare `<dt>`/`<dd>` groups or `<div>`-wrapped groups, never a mix, and `<dt>` forbids headings.

## Examples

### Basic Key Value Group

```tsx
<KeyValueGroup
  items={[
    { slotLabel: "Owner", slotValue: "Jordan Lee" },
    { slotLabel: "Region", slotValue: "us-west-2" },
    { slotLabel: "Created", slotValue: "15 Jan 2026" },
  ]}
/>
```

### Horizontal Key Value Group

Use a horizontal group for a compact row of stats, such as a summary strip above a table. Pairs wrap when the row runs out of room.

```tsx
<KeyValueGroup
  items={[
    { slotLabel: "Projects", slotValue: "8" },
    { slotLabel: "Storage", slotValue: "640 GB" },
    { slotLabel: "Plan", slotValue: "Team" },
  ]}
  orientation="horizontal"
/>
```

### Stacked Key Value Group

Use vertical pairs when the group sits in a narrow column, such as a sidebar, so each key stacks above its value.

```tsx
<KeyValueGroup
  items={[
    { slotLabel: "Owner", slotValue: "Jordan Lee" },
    { slotLabel: "Region", slotValue: "us-west-2" },
    { slotLabel: "Created", slotValue: "15 Jan 2026" },
  ]}
  pairOrientation="vertical"
/>
```

### Key Value Group With Rich Values

A value is any node. Pass a Badge for status and a read-only Tag for a category, and place an Icon or Avatar before the text. Icons hide themselves from assistive technology; mark the Avatar aria-hidden so the name is not announced twice.

```tsx
<KeyValueGroup
  items={[
    {
      slotLabel: "Status",
      slotValue: <Badge color="green">Active</Badge>,
    },
    {
      slotLabel: "Type",
      slotValue: (
        <Tag color="gray" kind="outline" readOnly>
          Internal
        </Tag>
      ),
    },
    {
      slotLabel: "Owner",
      slotValue: (
        <>
          <Avatar aria-hidden fallback="JL" size="small" />
          Jordan Lee
        </>
      ),
    },
    {
      slotLabel: "Updated",
      slotValue: (
        <>
          <Icon name="clock" />2 hours ago
        </>
      ),
    },
  ]}
/>
```

### Composed

```tsx
<KeyValueGroupRoot orientation="vertical">
  <KeyValueGroupItem orientation="horizontal">
    <KeyValueGroupKey>Owner</KeyValueGroupKey>
    <KeyValueGroupValue>
      <Avatar aria-hidden fallback="JL" size="small" />
      Jordan Lee
    </KeyValueGroupValue>
  </KeyValueGroupItem>
  <KeyValueGroupItem orientation="horizontal">
    <KeyValueGroupKey>Status</KeyValueGroupKey>
    <KeyValueGroupValue>
      <Badge color="green">Active</Badge>
    </KeyValueGroupValue>
  </KeyValueGroupItem>
  <KeyValueGroupItem orientation="vertical">
    <KeyValueGroupKey>Description</KeyValueGroupKey>
    <KeyValueGroupValue>Customer-facing web application for the online store.</KeyValueGroupValue>
  </KeyValueGroupItem>
</KeyValueGroupRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `KeyValueGroupItemData[]` | - | Pairs to render in order. |
| attributes | `{ KeyValueGroupItem?: NativeElementAttributes<"div", typeof KeyValueGroupItem>; KeyValueGroupKey?: NativeElementAttributes<"dt", typeof KeyValueGroupKey>; KeyValueGroupValue?: NativeElementAttributes<"dd", typeof KeyValueGroupValue> }` | - | Native HTML attributes forwarded to the internal composed components. |
| orientation | `"horizontal" \| "vertical"` | `"vertical"` | How the pairs flow. - "vertical" - Pairs stack. Horizontal pairs share one key column. - "horizontal" - Pairs sit side by side and wrap. |
| pairOrientation | `"horizontal" \| "vertical"` | `"horizontal"` | How each pair lays out its key and value. `orientation` controls how the pairs themselves flow. - "vertical" - Key above the value. - "horizontal" - Key beside the value. In a vertical group every key shares one column; in a horizontal group each key hugs its text. |

`* = required prop`

## Exports

- Components: `KeyValueGroup`, `KeyValueGroupItem`, `KeyValueGroupKey`, `KeyValueGroupRoot`, `KeyValueGroupValue`
- Types: `KeyValueGroupItemData`, `KeyValueGroupItemProps`, `KeyValueGroupKeyProps`, `KeyValueGroupProps`, `KeyValueGroupRootProps`, `KeyValueGroupValueProps`

## See also

- [Table](./Table.md)
- [List](./List.md)
