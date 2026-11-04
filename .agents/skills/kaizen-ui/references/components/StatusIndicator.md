# StatusIndicator

A small colored dot that signals a new change or short-lived status.

## Notes

- Pair it with adjacent text or an accessible label to convey the meaning beyond color alone.

- Use Badge (not StatusIndicator) when the status carries a text label. StatusIndicator is for the minimal dot-only case (table cells, list items, overlay on an avatar or icon).

- Follow the same color-to-status mapping as Badge so the meaning stays consistent across the app: green for active/healthy, red for error/offline, yellow for warning/degraded, blue for informational/in-progress.

## Examples

### Status Indicator Colors

Map the `color` prop to semantic meaning: red (the default) for error, yellow for warning, green for success or healthy, and blue for new/info. Also supports custom colors via CSS.

```tsx
<Flex direction="col" gap="2">
  <StatusIndicator color="red" />
  <StatusIndicator color="yellow" />
  <StatusIndicator color="green" />
  <StatusIndicator color="blue" />
  <StatusIndicator className="bg-accent-teal" />
</Flex>
```

### Status Indicator Sizes

Scale the `size` prop to match surrounding content — smaller for dense rows and inline use, larger for prominent dashboards or hero areas.

```tsx
<Flex align="center" gap="2">
  <StatusIndicator size="small" />
  <StatusIndicator size="medium" />
  <StatusIndicator size="large" />
  <StatusIndicator size="xlarge" />
  <StatusIndicator size="xxlarge" />
</Flex>
```

### Status Indicator On Icon

Anchor the indicator over an icon with a relatively positioned wrapper to flag unread or actionable items like notifications.

```tsx
<div className="relative w-fit">
  <Bell />
  <StatusIndicator className="absolute top-0 right-0" size="medium" />
</div>
```

### Status Indicator On Avatar

Overlay an avatar with a green indicator to communicate a user's online or active presence.

```tsx
<div className="relative w-fit">
  <Avatar fallback="NV" />
  <StatusIndicator className="absolute right-0.5 bottom-0.5 translate-x-1/2" color="green" size="medium" />
</div>
```

### Status Indicator With Label

Pair the indicator with an inline text label whenever the status must be unambiguous and accessible to assistive tech — color alone is not enough.

```tsx
<Flex align="center" gap="2">
  <StatusIndicator color="green" size="medium" />
  <span>Stable</span>
</Flex>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| color | `"blue" \| "green" \| "red" \| "yellow"` | `"red"` | Semantic color of the dot. |
| size | `"small" \| "medium" \| "large" \| "xlarge" \| "xxlarge"` | `"medium"` | Visual size of the dot. |

## Exports

- Components: `StatusIndicator`
- Types: `StatusIndicatorProps`

## See also

- [Badge](./Badge.md)
- [Tag](./Tag.md)
