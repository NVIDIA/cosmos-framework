# Block

A general-purpose block-level container for arbitrary content. Use it as a low-level layout primitive when consistent spacing and overflow behavior are needed without writing custom CSS.

## Notes

- Use Block for one region that needs padding, overflow control, or text truncation. Arrange several children with Stack, Flex, or Grid instead.

- Use `truncate` to clip a single line of text with an ellipsis. `textOverflow` on its own does nothing: it also needs `overflow` set to something other than `"visible"` and `textWrap="nowrap"`, which is what `truncate` sets.

- Prefer density tokens such as `padding="density-lg"` over fixed steps such as `"4"`, so spacing follows the density theme.

- Use `asChild` to apply Block's padding and overflow to a semantic element such as `<section>` or `<article>` instead of adding a wrapper `<div>`.

## Examples

### Basic Block

```tsx
<Block
  style={{
    height: "100px",
    width: "200px",
    border: "3px dotted var(--border-color-base)",
  }}
/>
```

## Props

Spacing values use the [spacing scale](../design-tokens/SPACING_SCALE.md).

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| gap | `SpacingScaleUnion` | - | Sets spacing between flex and grid items. |
| overflow | `"hidden" \| "auto" \| "clip" \| "scroll" \| "visible"` | - | Controls how content that overflows the box is handled on both axes. See https://developer.mozilla.org/en-US/docs/Web/CSS/overflow for more. - "auto" - Adds scrollbars only when content overflows. - "clip" - Clips overflow with no scrolling affordance. - "hidden" - Clips overflow and disables scrolling. - "scroll" - Always shows scrollbars. - "visible" - Renders overflow outside the box. |
| overflowX | `"hidden" \| "auto" \| "clip" \| "scroll" \| "visible"` | - | Controls how content that overflows the box is handled on the horizontal axis. See https://developer.mozilla.org/en-US/docs/Web/CSS/overflow-x for more. - "auto" - Adds a horizontal scrollbar only when content overflows. - "clip" - Clips horizontal overflow with no scrolling affordance. - "hidden" - Clips horizontal overflow and disables horizontal scrolling. - "scroll" - Always shows a horizontal scrollbar. - "visible" - Renders horizontal overflow outside the box. |
| overflowY | `"hidden" \| "auto" \| "clip" \| "scroll" \| "visible"` | - | Controls how content that overflows the box is handled on the vertical axis. See https://developer.mozilla.org/en-US/docs/Web/CSS/overflow-y for more. - "auto" - Adds a vertical scrollbar only when content overflows. - "clip" - Clips vertical overflow with no scrolling affordance. - "hidden" - Clips vertical overflow and disables vertical scrolling. - "scroll" - Always shows a vertical scrollbar. - "visible" - Renders vertical overflow outside the box. |
| padding | `SpacingScaleUnion` | - | Sets padding. |
| paddingBottom | `SpacingScaleUnion` | - | Sets bottom padding. |
| paddingLeft | `SpacingScaleUnion` | - | Sets left padding. |
| paddingRight | `SpacingScaleUnion` | - | Sets right padding. |
| paddingTop | `SpacingScaleUnion` | - | Sets top padding. |
| paddingX | `SpacingScaleUnion` | - | Sets horizontal padding. |
| paddingY | `SpacingScaleUnion` | - | Sets vertical padding. |
| textOverflow | `"clip" \| "ellipsis"` | - | Controls how inline text is signaled when it overflows the box. Requires `overflow` to be set to something other than `visible`. See https://developer.mozilla.org/en-US/docs/Web/CSS/text-overflow for more. - "ellipsis" - Renders an ellipsis (`…`) to indicate clipped text. - "clip" - Truncates the text at the edge of the content area. |
| textWrap | `"wrap" \| "nowrap" \| "balance" \| "pretty"` | - | Controls how text wraps across lines inside the block. See https://developer.mozilla.org/en-US/docs/Web/CSS/text-wrap for more. - "wrap" - Wraps text across as many lines as needed to fit the container. - "nowrap" - Keeps text on a single line; pair with `overflow` and `textOverflow` to clip. - "balance" - Distributes text evenly across lines, suited to short headings. - "pretty" - Optimizes line breaks for readability, suited to body copy. |
| truncate | `boolean` | - | Clips inline text to a single line and appends an ellipsis. Shorthand for `textWrap: "nowrap"`, `overflow: "hidden"`, and `textOverflow: "ellipsis"`. |

## Exports

- Components: `Block`
- Types: `BlockProps`

## See also

- [Stack](./Stack.md)
