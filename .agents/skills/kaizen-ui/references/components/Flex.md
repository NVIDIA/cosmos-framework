# Flex

A layout primitive that arranges its children with CSS flexbox using design-token spacing.

## Notes

- Flex defaults to a single row that does not wrap. Use Stack for a vertical sequence, and Grid when items must line up across both rows and columns.

- Set `wrap="wrap"` when a row of items, such as tags or buttons, can outgrow its container.

- Space children with `gap` instead of margins, and prefer density tokens such as `gap="density-md"` over fixed steps such as `"2"`, so spacing follows the density theme.

- Use `asChild` to lay out a semantic element such as `<ul>` or `<nav>` directly instead of nesting it in an extra `<div>`.

## Examples

### Basic Flex

Use a basic row layout when items should flow horizontally with consistent spacing.

```tsx
<Flex gap="2" padding="1" style={{ border: "1px dotted var(--border-color-base)" }}>
  <Flex padding="1" style={{ flex: 1, border: "1px dotted var(--border-color-base)" }}>
    Item A
  </Flex>
  <Flex padding="1" style={{ flex: 2, border: "1px dotted var(--border-color-base)" }}>
    Item B
  </Flex>
  <Flex padding="1" style={{ flex: 1, border: "1px dotted var(--border-color-base)" }}>
    Item C
  </Flex>
</Flex>
```

### Column Flex

Use a column layout with centered alignment when stacking items vertically within a constrained width.

```tsx
<Flex direction="col" align="center" gap="2">
  <div>Item A</div>
  <div>Item B</div>
  <div>Item C</div>
</Flex>
```

### Wrapping Flex

Use wrapping when items should flow to the next line rather than overflowing their container.

```tsx
<Flex gap="2" wrap="wrap">
  <div>Item A</div>
  <div>Item B</div>
  <div>Item C</div>
  <div>Item D</div>
  <div>Item E</div>
</Flex>
```

## Props

Spacing values use the [spacing scale](../design-tokens/SPACING_SCALE.md).

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| align | `"center" \| "start" \| "end" \| "baseline" \| "stretch"` | `"stretch"` | Alignment of items along the cross axis. Maps to CSS `align-items`. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| direction | `"col" \| "row" \| "column" \| "row-reverse" \| "col-reverse" \| "column-reverse"` | `"row"` | Direction items flow along the main axis. Maps to CSS `flex-direction`. |
| gap | `SpacingScaleUnion` | - | Sets spacing between flex and grid items. |
| justify | `"center" \| "start" \| "end" \| "normal" \| "stretch" \| "between" \| "around" \| "evenly"` | `"start"` | Distribution of space along the main axis. Maps to CSS `justify-content`. |
| padding | `SpacingScaleUnion` | - | Sets padding. |
| paddingBottom | `SpacingScaleUnion` | - | Sets bottom padding. |
| paddingLeft | `SpacingScaleUnion` | - | Sets left padding. |
| paddingRight | `SpacingScaleUnion` | - | Sets right padding. |
| paddingTop | `SpacingScaleUnion` | - | Sets top padding. |
| paddingX | `SpacingScaleUnion` | - | Sets horizontal padding. |
| paddingY | `SpacingScaleUnion` | - | Sets vertical padding. |
| wrap | `"wrap" \| "nowrap" \| "wrap-reverse"` | `"nowrap"` | Wrapping behavior of the container. Maps to CSS `flex-wrap`. |

## Exports

- Components: `Flex`
- Types: `FlexProps`

## See also

- [Stack](./Stack.md)
- [Grid](./Grid.md)
