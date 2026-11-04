# Inline

An inline layout primitive that renders a `<span>` with the system's spacing tokens exposed as props.

## Notes

- Use Inline to pad a short run of content inside a line of text, such as a highlighted term. Use Block when the content should start its own line.

- Inline has padding props only, with no `gap` or alignment props; use Flex to space or align a group of items.

- Vertical padding on Inline enlarges its background but does not push neighboring lines apart, so use Block when vertical space must affect layout.

- Use `asChild` to pad a semantic inline element such as `<code>` or `<a>` instead of wrapping it in an extra `<span>`.

## Examples

### Basic Inline

```tsx
<Inline padding="2" style={{ border: "1px dotted var(--border-color-base)" }}>
  Inline text
</Inline>
```

## Props

Spacing values use the [spacing scale](../design-tokens/SPACING_SCALE.md).

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| padding | `SpacingScaleUnion` | - | Sets padding. |
| paddingBottom | `SpacingScaleUnion` | - | Sets bottom padding. |
| paddingLeft | `SpacingScaleUnion` | - | Sets left padding. |
| paddingRight | `SpacingScaleUnion` | - | Sets right padding. |
| paddingTop | `SpacingScaleUnion` | - | Sets top padding. |
| paddingX | `SpacingScaleUnion` | - | Sets horizontal padding. |
| paddingY | `SpacingScaleUnion` | - | Sets vertical padding. |

## Exports

- Components: `Inline`
- Types: `InlineProps`

## See also

- [Block](./Block.md)
