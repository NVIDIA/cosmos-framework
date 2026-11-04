# Badge

A small color-coded label that displays status, category, or metadata at a glance. Badges are
non-interactive and convey information rather than trigger actions.

## Notes

- Always use Badge (not plain text) to display entity status. Pick a semantic color (e.g. color="green" for healthy, color="yellow" for degraded).

- Default to kind="solid" almost everywhere — the solid treatment carries the status signal at a glance. Reach for kind="outline" only in deliberate de-emphasized contexts

- Pair status badges with an icon that reinforces the semantics (check for healthy, warning triangle for degraded, etc.). The icon aids color-blind users and lets the meaning survive at a glance even before the label is read.

- Use Tag (not Badge) for interactive labels (selectable, dismissible, clickable) and for taxonomic categories (topic tags, tech stack, team name, region).

- Use StatusIndicator (not Badge) when the status needs only a colored dot without a text label.

- Establish a single project-wide color-to-status mapping (e.g. green = healthy, red = error, yellow = warning) and use it consistently — do not reuse the same color for different meanings on the same page.

- Placement: place status badges near the title (top-right of a card, inline with a detail-page title) and never inside a tag row (tags read as interactive, badges do not). If a card has no status, omit the Badge rather than inserting a placeholder for grid symmetry.

- Limit to 2-3 badges per entity; excess badges become visual noise users ignore.

- In dense table cells, prefer plain text or `Anchor` over Badge/Tag for taxonomic labels.

- Prefer no icons - status badges can use color and text to convey meaning. Reserve icons for badges that need to scan at very small sizes, or to disambiguate two states with the same color.

- Avoid rainbow-coding - use a single color for each status.

- Set `kind="solid"` for badges in detail views, side panels, and status bars - it provides clear, immediate visual hierarchy for status. `kind="outline"` is visually distracting when it appears alongside other bordered elements(panels, inputs, dividers).

- Prefer `kind="outline"`(default) when the badge is **secondary** information that should not compete for attention - eg card metadata, labels in a dense list where multiple badges appear per row

## Examples

### Basic Badge

Pick `kind="solid"` for status badges in detail views, side panels, and status bars. Use `kind="outline"` only for de-emphasized contexts like card metadata — outline badges visually resemble Tags and Buttons, creating a false interactive affordance.

```tsx
<Grid cols={2} gap="3" style={{ placeItems: "center" }}>
  <Badge color="blue" kind="solid">
    Badge Text
  </Badge>
  <Badge color="blue">Badge Text</Badge>
  <Badge color="green" kind="solid">
    Badge Text
  </Badge>
  <Badge color="green">Badge Text</Badge>
  <Badge color="yellow" kind="solid">
    Badge Text
  </Badge>
  <Badge color="yellow">Badge Text</Badge>
  <Badge color="red" kind="solid">
    Badge Text
  </Badge>
  <Badge color="red">Badge Text</Badge>
</Grid>
```

### Status Badge

Map `color` to a single project-wide status meaning (green = healthy, red = error, yellow = warning, blue = informational, gray = inactive) and reuse it across the app. Do not assign the same color to different meanings on the same page.

```tsx
<Flex gap="density-md" wrap="wrap">
  <Badge color="green" kind="solid">
    Active
  </Badge>
  <Badge color="red" kind="solid">
    Failed
  </Badge>
  <Badge color="yellow" kind="solid">
    Warning
  </Badge>
  <Badge color="blue" kind="solid">
    In Progress
  </Badge>
  <Badge color="gray" kind="solid">
    Inactive
  </Badge>
</Flex>
```

### With Icon

Pair an icon with the label when the status benefits from redundant encoding (especially for color-blind accessibility). Omit icons for category badges where color alone is enough — extra glyphs add visual noise in dense layouts.

```tsx
<Badge color="purple">
  <Shield /> Encrypted
</Badge>
```

### As Link

```tsx
<Badge asChild>
  <a href="/">Badge as a Link</a>
</Badge>
```

### With Truncation

Badge text should be short and concise. By default, overflowing content wraps so it stays readable. If you must truncate (e.g., a constrained layout with unpredictable content), wrap the text in a span with truncation styles.

```tsx
<Badge>
  <Shield />
  <span className="truncate">Long text that will be truncated</span>
</Badge>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| color | `"blue" \| "green" \| "red" \| "yellow" \| "purple" \| "teal" \| "gray"` | `"blue"` | Semantic color used to convey meaning at a glance. |
| kind | `"outline" \| "solid"` | `"outline"` | Visual treatment of the badge. - "outline" - De-emphasized contexts such as card metadata or secondary information. - "solid" - Prominent status badges in detail views, side panels, and status bars. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Controls the badge's padding and text size. Reach for `small` in very dense layouts, and `large` for prominent status in detail views. |

## Exports

- Components: `Badge`
- Types: `BadgeProps`

## See also

- [Tag](./Tag.md)
- [StatusIndicator](./StatusIndicator.md)
