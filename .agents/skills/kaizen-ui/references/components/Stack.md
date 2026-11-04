# Stack

A flex container that arranges its children in a column by default, with an optional divider between items.

## Notes

- Prefer Stack over Flex for sequential layouts — Stack defaults to a column and communicates intent more clearly.

- Use `slotDivider` to inject separators between children rather than rendering dividers manually between items.

## Examples

### Basic Stack

```tsx
<Stack padding="1" gap="2" style={{ border: "3px dotted var(--border-color-base)" }}>
  <div>Item A</div>
  <div>Item B</div>
  <div>Item C</div>
</Stack>
```

### Horizontal Stack

```tsx
<Stack direction="row" gap="2">
  <div>Item A</div>
  <div>Item B</div>
  <div>Item C</div>
</Stack>
```

### With Divider Stack

Use slotDivider to automatically inject a separator between each child without manual repetition.

```tsx
<Stack gap="2" slotDivider={<Divider />}>
  <div>Section A</div>
  <div>Section B</div>
  <div>Section C</div>
</Stack>
```

### Aligned Stack

Use align to position children along the cross-axis (e.g. center) when items have varying widths in a column or heights in a row.

```tsx
<Stack gap="2" align="center">
  <div>Short</div>
  <div>A longer item</div>
  <div>Mid</div>
</Stack>
```

### Justified Stack

Use direction=row with justify=between to push siblings to opposite ends of a bar, the canonical pattern for toolbars and page headers.

```tsx
<Stack direction="row" justify="between">
  <div>Leading</div>
  <div>Trailing</div>
</Stack>
```

## Props

Spacing values use the [spacing scale](../design-tokens/SPACING_SCALE.md).

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| align | `"center" \| "start" \| "end" \| "baseline" \| "stretch"` | `"stretch"` | Alignment of items along the cross axis. Maps to CSS `align-items`. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| direction | `"col" \| "row" \| "column" \| "row-reverse" \| "col-reverse" \| "column-reverse"` | `"col"` | Direction items flow along the main axis. Maps to CSS `flex-direction`. |
| gap | `SpacingScaleUnion` | - | Sets spacing between flex and grid items. |
| justify | `"center" \| "start" \| "end" \| "normal" \| "stretch" \| "between" \| "around" \| "evenly"` | `"start"` | Distribution of space along the main axis. Maps to CSS `justify-content`. |
| padding | `SpacingScaleUnion` | - | Sets padding. |
| paddingBottom | `SpacingScaleUnion` | - | Sets bottom padding. |
| paddingLeft | `SpacingScaleUnion` | - | Sets left padding. |
| paddingRight | `SpacingScaleUnion` | - | Sets right padding. |
| paddingTop | `SpacingScaleUnion` | - | Sets top padding. |
| paddingX | `SpacingScaleUnion` | - | Sets horizontal padding. |
| paddingY | `SpacingScaleUnion` | - | Sets vertical padding. |
| slotDivider | `ReactNode` | - | Element rendered between each pair of children. |
| wrap | `"wrap" \| "nowrap" \| "wrap-reverse"` | `"nowrap"` | Wrapping behavior of the container. Maps to CSS `flex-wrap`. |

## Exports

- Components: `Stack`
- Types: `StackProps`

## See also

- [Divider](./Divider.md)
