# Tag

A small label that categorizes or groups an item with text and optional icons. Use for classification and filtering; reach for Badge to communicate status.

## Notes

- Tags should be label-only by default. Do not add icons unless the tag represents a typed entity (e.g., a person tag with an avatar).

- Do not use Tag for multi-value table cells in dense data tables — pills compound into visual noise. Use plain text or Anchor components separated by commas instead.

- Use Badge (not Tag) for read-only status or category labels — Badge has no interaction states. Reach for Tag when the label is selectable, dismissible, or clickable.

- Even though the framework default is `color="blue"`, pass `color="gray"` explicitly when the tag has no semantic meaning — a uniform neutral palette across the application keeps the status-vs-category signal intact. Semantic colors (green/red/yellow) are reserved for Badge.

- Prefer `kind="outline"` for most Tag usage — the outline treatment visually distinguishes Tags (interactive) from solid Badges (read-only status).

- Tag is interactive (filter chip, selectable label, typed entity) while Badge is read-only status. Never nest a Badge inside a Tag row, and never put a Tag inside a status slot.

- Rainbow-coding anti-pattern: per-value maps like `TAG_COLOR = { gpu: "green", cpu: "blue" }` collapse the color-as-status signal. Keep tag colors neutral (`color="gray"`) and use `kind="solid"` vs `kind="outline"` to distinguish entity types (e.g. Model vs Blueprint vs Service).

- If you make Tags dismissible, pair the chip with a clear re-add path (e.g. a filter list, "Add filter" menu). Dismissing without a way back leaves users unable to recover removed filters.

## Examples

### Basic Tag

Default Tags to a single neutral color (`gray`) across the application. Use `kind="outline"` when you need a lighter visual weight, such as in dense layouts or alongside solid tags for hierarchy.

```tsx
<Grid cols={2} gap="3" style={{ placeItems: "center" }}>
  <Tag color="gray">Tag Text</Tag>
  <Tag color="gray" kind="outline">
    Tag Text
  </Tag>
  <Tag color="gray" disabled>
    Tag Text
  </Tag>
  <Tag color="gray" disabled kind="outline">
    Tag Text
  </Tag>
</Grid>
```

### With Icon Tag

Tags are label-only by default. Reserve a leading icon for typed-entity tags (e.g. a person tag with an avatar) where the icon identifies what kind of thing the tag represents.

```tsx
<Tag>
  <Icon />
  Tag Text
</Tag>
```

### Selected Tag

Use to indicate the tag is actively selected in a filter or multi-select context. Automatically triggered by children that are `:checked` or `[data-state="checked"]` - use a visually hidden (`nv-sr-only`) checkbox or radio input so the control stays keyboard-focusable while controlling the state.

```tsx
<Flex direction="col" gap="density-md">
  <Tag selected>Selected</Tag>
  <Tag asChild>
    <label>
      <input className="nv-sr-only" type="checkbox" /> Checkbox
    </label>
  </Tag>
  <Tag asChild>
    <label>
      <input className="nv-sr-only" type="radio" /> Radio
    </label>
  </Tag>
</Flex>
```

### Disabled Tag

Use to communicate that an interactive tag is temporarily unavailable; pair with explanatory copy when the reason is not obvious.

```tsx
<Tag disabled>Disabled</Tag>
```

### Read Only Tag

Use when the tag is informational only and should not be interactive, such as metadata labels. Tag will render as a `span` instead of a `button`.

```tsx
<Tag readOnly>Read Only</Tag>
```

### Colors Tag

Tags should default to a single neutral color (`gray`) across a collection. The non-neutral example here is user-applied metadata — the user explicitly chose `purple` when tagging the resource. Do not assign colors per categorical value (rainbow coding); semantic colors are reserved for Badge.

```tsx
<Flex gap="density-md" wrap="wrap">
  <Tag color="gray">Project Alpha</Tag>
  <Tag color="purple">Project Alpha</Tag>
</Flex>
```

### Density Tag

Use `compact` for dense data tables and filter bars, `spacious` for prominent filter chips or hero surfaces, and the default `standard` for general use. Note that density is also inherited from the parent container.

```tsx
<Flex align="center" gap="density-md">
  <Tag density="compact">Compact</Tag>
  <Tag>Standard</Tag>
  <Tag density="spacious">Spacious</Tag>
</Flex>
```

### Dismissible Tag

Use a single tag with a trailing close icon when clicking anywhere on the tag should remove it, such as a list of applied filter chips.

```tsx
<Tag aria-label="Remove filter">
  Filter
  <CloseIcon />
</Tag>
```

### Split Tag

Wrap two adjacent tags in `Group` when the label and a secondary action (e.g. dismiss, edit) need separate click targets.

```tsx
<Group>
  <Tag>Filter</Tag>
  <Tag aria-label="Remove filter">
    <CloseIcon />
  </Tag>
</Group>
```

### As Link Tag

Use `asChild` with an anchor when the tag should navigate; the anchor inherits tag styling while staying keyboard-accessible.

```tsx
<Tag asChild>
  <a href="/">Tag as Link</a>
</Tag>
```

### Truncated Tag

Tags wrap their content by default. Only wrap text in a `truncate` span when the tag width is constrained and wrapping is unacceptable; pair with a Tooltip so the full label stays reachable.

```tsx
<Tag title="Tag content that may be too long">
  <Icon />
  <span className="truncate">Tag content that may be too long</span>
</Tag>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| color | `"blue" \| "green" \| "red" \| "yellow" \| "purple" \| "teal" \| "gray"` | `"blue"` | Accent color of the tag. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces. |
| kind | `"outline" \| "solid"` | `"solid"` | The visual style of the tag. |
| readOnly | `boolean` | - | Renders the tag as a non-interactive `<span>` instead of a `<button>`. Use when placing a tag inside another interactive element like a clickable card. |
| selected | `boolean` | - | Marks the tag as selected, applying the selected style and removing the hover state. |

## Exports

- Components: `Tag`
- Types: `TagProps`

## See also

- [Badge](./Badge.md)
- [Group](./Group.md)
