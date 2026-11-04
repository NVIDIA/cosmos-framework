# Skeleton

A placeholder block that mimics the shape of content while it loads. Match each skeleton's size and shape to the real content as closely as possible to keep layout shift minimal.

## Notes

- Skeleton works because mirroring the expected layout reduces perceived load time — that is the core reason to prefer Skeleton over Spinner when the layout is known.

- Use Spinner (not Skeleton) when the layout of the loading content is unknown. Skeleton only pays off when its shapes mirror the real content.

- To avoid layout shift, wrap the real content or representative placeholder text in Skeleton and toggle `loading`, e.g. `<Skeleton loading={isLoading}><Badge>...</Badge></Skeleton>`. Prefer this over `kind="pill"` for badges and tags.

- Put `aria-busy="true"` on the region that is loading; Skeleton hides its own placeholder content from assistive technology.

- Use ProgressBar (not Skeleton) when the operation has a measurable percentage.

- Skip the skeleton for very fast loads (under ~200ms) — it flickers in and out and adds visual noise without reducing perceived load time.

## Examples

### Kinds Skeleton

Pick a `kind` to match the shape of the content being loaded: `line` (default) for single-line text, `pill` for tags or badges, and `circle` for avatars or icons.

```tsx
<Stack gap="density-md">
  <Skeleton kind="circle" />
  <Skeleton kind="pill" />
  <Skeleton kind="line" />
  <Skeleton kind="line" />
</Stack>
```

### Static Skeleton

Set `animated={false}` when the skeleton flashes briefly or sits inside a surface that already conveys a loading state, where the pulse would be visual noise.

```tsx
<Skeleton kind="line" animated={false} />
```

### Inline Text Skeleton

A `line` skeleton is exactly one line of the surrounding text tall, so it can stand in for a heading or label without shifting the layout when the text loads.

```tsx
<Text kind="title/lg">
  <Skeleton />
</Text>
```

### Sized Skeleton

Override the default dimensions with `className` or `style` so the skeleton matches the exact size of the component it stands in for, like a large `Avatar` or icon.

```tsx
<Skeleton kind="circle" className="size-12" />
```

### Content Block Skeleton

Compose multiple skeleton shapes inside layout primitives to mirror the rough size and rhythm of a content block so the page does not jump when real data loads in.

```tsx
<Stack gap="density-lg" style={{ maxWidth: "320px" }}>
  <Skeleton kind="circle" />
  <Skeleton kind="line" style={{ width: "66%" }} />
  <Stack gap="density-sm">
    <Skeleton kind="line" />
    <Skeleton kind="line" />
    <Skeleton kind="line" />
  </Stack>
  <Flex gap="density-sm" wrap="wrap">
    <Skeleton kind="pill" />
    <Skeleton kind="pill" />
  </Flex>
</Stack>
```

### Wrapped Content Skeleton

Wrap the real component and toggle `loading` to cover it in place. The skeleton takes the component's exact size and shape, so nothing moves when it loads. Prefer this to `kind="pill"` for badges and tags.

```tsx
<Flex align="center" gap="density-md">
  <Skeleton>
    <Avatar fallback="JD" />
  </Skeleton>
  <Skeleton loading>
    <Badge color="green" kind="solid">
      Active
    </Badge>
  </Skeleton>
</Flex>
```

### Multiline Text Skeleton

Pass representative text to draw one bar per wrapped line. The text is laid out but invisible, so the paragraph keeps its real height while it loads.

```tsx
<Text kind="body/regular/md" style={{ maxWidth: "280px" }}>
  <Skeleton>Placeholder copy about as long as the description that loads here, so it wraps onto the same number of lines.</Skeleton>
</Text>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| animated | `boolean` | `true` | Whether the skeleton pulses while visible. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| children | `ReactNode` | - | Content to size the placeholder from. Text wraps into one bar per line, and a single element is covered in place, so the layout does not shift when `loading` turns `false`. A single element must pass `className` and other DOM props to its root, as every KUI component does. While loading, the children are hidden from assistive technology and cannot be focused or clicked. |
| kind | `"circle" \| "line" \| "pill"` | `"line"` | Shape of the placeholder when it has no children. Pick the option that most closely matches the content it stands in for. - "line" - Text-shaped block exactly one line of the surrounding text tall. - "pill" - Fixed-width rounded rectangle for tag- or chip-like content. - "circle" - Fixed-diameter circle for avatars and icons.  Ignored when `children` are passed, because the skeleton then takes the shape of its children. |
| loading | `boolean` | `true` | Whether to show the placeholder. When `false`, the skeleton renders its children as-is, or nothing when it has no children or uses `asChild`. |

## Exports

- Components: `Skeleton`
- Types: `SkeletonProps`
