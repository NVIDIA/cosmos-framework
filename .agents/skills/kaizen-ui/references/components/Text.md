# Text

Renders text with the design system's typography tokens applied.

## Notes

- Always pick a `kind` from the typography scale rather than overriding `fontSize` / `fontWeight` individually.

- Pair heading kinds (`title/*`, `display/*`) with `asChild` and a semantic `h1`–`h6` so screen readers can navigate by heading level.

- Do not render text in uppercase — no `uppercase` utility, no manually capitalized strings. Acronyms (API, GPU, URL) are the only exception.

- Text renders an inline `<span>`, so vertical margin utilities (`mt-*`, `mb-*`, `my-*`) are silently dropped — horizontal margins (`ml-*`, `mr-*`, `mx-*`) work fine. Inside a `Flex` or `Stack` this rarely surfaces because the parent `gap` handles spacing; it bites standalone `Text` in a vertical flow, where you should add `block` (or wrap in a block element).

- Do not hallucinate `kind` values — only use the ones defined in the type.

- When in doubt, map context to kind: page heading → `title/lg`; section heading → `title/md`; sub-section heading → `title/sm`; paragraph → `body/regular/md`; card title → `body/bold/xl`; card description → `body/regular/sm`; metadata → `label/regular/sm` with `text-secondary`; form label → `label/regular/sm`; hero headline → `display/lg`; code → `mono/sm` or `mono/md`.

- Do NOT render text in uppercase because all-caps reduces reading speed 10-15%

- `display/*` kinds are reserved for hero and marketing surfaces; use `title/*` for standard in-app headings.

## Examples

### Kinds Text

Pick a `kind` from one of the five families: `display/*` for the largest hero text, `title/*` for headings, `body/*` for prose, `label/*` for short UI labels, and `mono/*` for code or technical identifiers. Each family offers multiple sizes and weights — see the `kind` type for the full set.

```tsx
<Stack gap="density-md">
  <Text kind="display/lg">Display Heading</Text>
  <Text kind="title/md">Section Title</Text>
  <Text kind="body/regular/md">The quick brown fox jumps over the lazy dog</Text>
  <Text kind="label/semibold/sm">Form Label</Text>
  <Text kind="mono/md">console.log("hello")</Text>
</Stack>
```

### Custom Styled Text

Reach for the granular fontFamily, fontWeight, fontSize, lineHeight, and fontStyle props as an escape hatch when no predefined kind matches the design.

```tsx
<Text fontFamily="sans" fontWeight="semibold" fontSize="18" lineHeight="150" fontStyle="italic">
  Custom typography
</Text>
```

### Underlined Text

Add the underline prop on top of any kind to flag inline emphasis like links or highlighted terms without changing the type style.

```tsx
<Text kind="body/regular/md" underline>
  Underlined passage
</Text>
```

### Inherit Kind Text

Use kind="inherit" when Text is nested inside an element that already defines the type style (e.g. inside another Text or a styled heading) and you only need the Text primitive's other props.

```tsx
<Text kind="inherit">Inherits parent typography</Text>
```

### As Child Text

Use asChild to apply Text styling to a semantic HTML element like h1–h6 without adding an extra wrapper.

```tsx
<Text asChild kind="title/lg">
  <h1>Semantic Heading</h1>
</Text>
```

## Props

Typography values use the [typography scale](../design-tokens/TYPOGRAPHY.md).

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| fontFamily | `"sans" \| "mono"` | - | Overrides the font family inherited from `kind`. |
| fontSize | `"10" \| "12" \| "14" \| "16" \| "18" \| "20" \| "22" \| "24" \| "28" \| "32" \| "36" \| "40" \| "44" \| "48" \| "50" \| "56" \| "60" \| "64" \| "72" \| "80"` | - | Overrides the font size inherited from `kind`, in pixels. |
| fontStyle | `"normal" \| "italic"` | - | Sets the font style. |
| fontWeight | `"bold" \| "light" \| "regular" \| "semibold"` | - | Overrides the font weight inherited from `kind`. |
| kind | `TypographyKindUnion` | `"label/regular/md"` | A semantic typography token combining family, weight, and size. Pass "inherit" to keep the parent's text style. Use "display" for the largest hero text, "title" for headings, "body" for paragraphs, "label" for short labels and UI text, and "mono" for code or technical content. |
| lineHeight | `"100" \| "125" \| "150" \| "175"` | - | Overrides the line height inherited from `kind`, as a percentage of the font size. |
| underline | `boolean` | - | Applies an underline to the text. |

## Exports

- Components: `Text`
- Types: `TextProps`
