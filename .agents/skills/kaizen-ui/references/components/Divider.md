# Divider

A horizontal or vertical line that visually separates groups of content. Use to create breaks between sections, list items, or toolbar regions where a heading or whitespace alone would be too subtle.

## Notes

- Prefer spacing or a section heading over a Divider inside cards and between form sections — lines there read as visual noise. Reserve Divider for breaks between distinct regions (sidebar sections, list groups, panel areas). For form sections, replace dividers with a `Text kind="title/sm"` heading and spacing.

- Use `orientation="vertical"` only for toolbars or side-by-side layouts; default to horizontal.

- Default to `width="small"`; reserve `"medium"` for more visual weight when separating major page sections.

## Examples

### Basic Divider

```tsx
<Flex className="w-full" direction="col" gap="3">
  <Divider />
  <Divider width="medium" />
  <Divider width="large" />
</Flex>
```

### With Text

```tsx
<Flex align="center" gap="density-lg">
  <Divider />
  <Text kind="label/semibold/md">Text</Text>
  <Divider />
</Flex>
```

### Vertical Divider

```tsx
<Divider orientation="vertical" />
```

### Composed

```tsx
<DividerRoot paddingY="2">
  <DividerElement orientation="horizontal" width="medium" />
</DividerRoot>
```

## Props

Spacing values use the [spacing scale](../design-tokens/SPACING_SCALE.md).

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ DividerRoot?: NativeElementAttributes<"div", typeof DividerRoot>; DividerElement?: NativeElementAttributes<"div", typeof DividerElement> }` | - | Native HTML attributes forwarded to the internal composed components. |
| orientation | `"horizontal" \| "vertical"` | `"horizontal"` | Axis the separator runs along. |
| padding | `SpacingScaleUnion` | - | Sets padding. |
| paddingBottom | `SpacingScaleUnion` | - | Sets bottom padding. |
| paddingLeft | `SpacingScaleUnion` | - | Sets left padding. |
| paddingRight | `SpacingScaleUnion` | - | Sets right padding. |
| paddingTop | `SpacingScaleUnion` | - | Sets top padding. |
| paddingX | `SpacingScaleUnion` | - | Sets horizontal padding. |
| paddingY | `SpacingScaleUnion` | - | Sets vertical padding. |
| width | `"small" \| "medium" \| "large"` | `"small"` | Thickness of the separator line. Step up to `"medium"` only when separating major page sections that need extra visual weight |

## Exports

- Components: `Divider`, `DividerElement`, `DividerRoot`
- Types: `DividerElementProps`, `DividerProps`, `DividerRootProps`

## See also

- [Stack](./Stack.md)
