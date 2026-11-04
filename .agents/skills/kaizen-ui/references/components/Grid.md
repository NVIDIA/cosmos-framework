# Grid

A two-dimensional layout primitive that arranges children into rows and columns. Use for dashboards, card listings, and any layout that needs both horizontal and vertical alignment.

## Notes

- Reach for `colMinWidth` over `cols` when items should reflow with the viewport — the browser sizes columns to fit the available space.

- Prefer the density/spacing tokens from the `gap` and `padding` enums i.e. "density-md" over "2"

## Examples

### Basic Grid

Use auto-flowing columns for responsive layouts where items should wrap based on available space.

```tsx
<Grid colMinWidth="250px" gap="4">
  <div>Item 1</div>
  <div>Item 2</div>
  <div>Item 3</div>
</Grid>
```

### Fixed Columns Grid

Use a fixed column count when the layout should not reflow regardless of viewport width.

```tsx
<Grid cols={3} gap="4">
  <div>Column A</div>
  <div>Column B</div>
  <div>Column C</div>
</Grid>
```

### Responsive Columns Grid

Use responsive breakpoint columns when you need explicit control over column counts at each viewport size. Autoflowing columns is usually preferred over this.

```tsx
<Grid cols={{ sm: 1, md: 2, lg: 3 }} gap="4">
  <div>Column A</div>
  <div>Column B</div>
  <div>Column C</div>
</Grid>
```

### Static Columns Grid

Pair `colMinWidth` with `colBehavior="static"` for uniform card grids where the card does not grow beyond the minimum width.

```tsx
<Grid colMinWidth="200px" colBehavior="static" gap="4">
  <div>Card 1</div>
  <div>Card 2</div>
  <div>Card 3</div>
</Grid>
```

### Column Flow Grid

Combine `rows` with `flow="col"` for column-major layouts where items fill top-to-bottom before wrapping to the next column.

```tsx
<Grid rows={3} flow="col" gap="2">
  <div>1</div>
  <div>2</div>
  <div>3</div>
  <div>4</div>
  <div>5</div>
  <div>6</div>
</Grid>
```

### Dense Flow Grid

Use `flow="dense"` with mixed `GridItem` spans to backfill gaps left by larger items, producing a tightly packed masonry-style layout.

```tsx
<Grid cols={4} flow="dense" gap="2">
  <GridItem cols={2}>Wide A</GridItem>
  <GridItem>B</GridItem>
  <GridItem rows={2}>Tall C</GridItem>
  <GridItem cols={3}>Wide D</GridItem>
  <GridItem>E</GridItem>
  <GridItem>F</GridItem>
</Grid>
```

### Composed

Use GridItem when individual cells need to span multiple columns or be placed at specific grid positions.

```tsx
<Grid cols={3} gap="4">
  <GridItem cols={2}>Spanning two columns</GridItem>
  <GridItem>Single column</GridItem>
  <GridItem colStart={2} colEnd={4}>
    Columns 2–3
  </GridItem>
</Grid>
```

## Props

Spacing values use the [spacing scale](../design-tokens/SPACING_SCALE.md).

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| colBehavior | `"grow" \| "static"` | `"grow"` | How auto-flowing columns fill leftover space when `colMinWidth` is set. - `"grow"`: Columns expand to fill available space when there aren't enough items to fill a row - `"static"`: Columns maintain their minimum width, leaving empty space if needed |
| colMinWidth | `string \| number` | - | Minimum column width that triggers responsive auto-flowing columns, sized by the browser to fit the available space. Accepts any CSS length, or numbers which are interpreted as pixels. Takes precedence over `cols` when set. |
| cols | `number \| { base?: number; xs?: number; sm?: number; md?: number; lg?: number; xl?: number; xxl?: number; }` | - | Number of fixed columns in the grid. Ignored when `colMinWidth` is set. |
| flow | `string` | - | CSS `grid-auto-flow` value controlling how items are placed into implicit tracks. |
| gap | `SpacingScaleUnion` | - | Sets spacing between flex and grid items. |
| padding | `SpacingScaleUnion` | - | Sets padding. |
| paddingBottom | `SpacingScaleUnion` | - | Sets bottom padding. |
| paddingLeft | `SpacingScaleUnion` | - | Sets left padding. |
| paddingRight | `SpacingScaleUnion` | - | Sets right padding. |
| paddingTop | `SpacingScaleUnion` | - | Sets top padding. |
| paddingX | `SpacingScaleUnion` | - | Sets horizontal padding. |
| paddingY | `SpacingScaleUnion` | - | Sets vertical padding. |
| rows | `number \| { base?: number; xs?: number; sm?: number; md?: number; lg?: number; xl?: number; xxl?: number; }` | - | Number of fixed rows in the grid. |

## Exports

- Components: `Grid`, `GridItem`
- Types: `GridItemLayout`, `GridItemProps`, `GridProps`, `LineValue`, `OneToTwelve`, `SpanValue`
