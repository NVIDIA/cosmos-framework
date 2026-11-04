# Anchor

Also known as: Link

Interactive text that navigates the user to another page, section, or resource.

## Notes

- Use Anchor for navigation (URL or route). For actions that do not navigate (submit, toggle, open a modal), use Button instead.

- To open a link in a new tab, use `external` rather than setting `target="_blank"` and `rel` by hand — it also adds the external-link icon and the accessible suffix.

## Examples

### Basic Anchor

```tsx
<Flex gap="2">
  <Anchor href="/" kind="inline">
    Inline
  </Anchor>
  <Anchor href="/" kind="standalone">
    Standalone
  </Anchor>
</Flex>
```

### Custom Link Component

Use asChild to render framework specific link components with our styling

```tsx
<Anchor asChild>
  <NextLink href="/">Home</NextLink>
</Anchor>
```

### External Link Anchor

Use `external` to open the link in a new tab, secure it with `rel="noopener noreferrer"`, and append an external-link icon plus an accessible "(opens in a new tab)" suffix, all in one prop.

```tsx
<Anchor href="https://example.com" external>
  Open documentation
</Anchor>
```

### Disabled Anchor

A disabled anchor will render as a span instead of an anchor tag

```tsx
<Anchor disabled href="/">
  Disabled Link
</Anchor>
```

### As Button

For an Anchor that looks like a link but triggers an action

```tsx
<Anchor asChild>
  <button onClick={() => console.log("Button clicked")} type="button">
    View Details
  </button>
</Anchor>
```

## Props

Typography values use the [typography scale](../design-tokens/TYPOGRAPHY.md).

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| disabled | `boolean` | `false` | Renders the anchor as a non-interactive `<span>` so it no longer navigates. |
| external | `boolean` | `false` | Opens the link in a new tab, secures it with `rel`, and appends an external-link icon and a screen-reader-only "(opens in a new tab)" suffix. An explicit `target` or `rel` still wins, and the suffix is dropped if that target is not `_blank`. Ignored when `disabled`. |
| fontFamily | `"sans" \| "mono"` | - | Overrides the font family inherited from `kind`. |
| fontSize | `"10" \| "12" \| "14" \| "16" \| "18" \| "20" \| "22" \| "24" \| "28" \| "32" \| "36" \| "40" \| "44" \| "48" \| "50" \| "56" \| "60" \| "64" \| "72" \| "80"` | - | Overrides the font size inherited from `kind`, in pixels. |
| fontStyle | `"normal" \| "italic"` | - | Sets the font style. |
| fontWeight | `"bold" \| "light" \| "regular" \| "semibold"` | - | Overrides the font weight inherited from `kind`. |
| kind | `"inline" \| "standalone"` | `"inline"` | The kind of anchor. - "inline" - Embedded within prose; renders with an underline so the link remains distinguishable inside body text. - "standalone" - Rendered outside of prose, such as a navigation link or call to action; renders without an underline. |
| lineHeight | `"100" \| "125" \| "150" \| "175"` | - | Overrides the line height inherited from `kind`, as a percentage of the font size. |
| textKind | `TypographyKindUnion` | `"body/regular/md"` | Typography token applied to the link text. |
| underline | `boolean` | - | Applies an underline to the text. |

## Exports

- Components: `Anchor`
- Types: `AnchorProps`

## See also

- [Button](./Button.md)
- [Breadcrumbs](./Breadcrumbs.md)
