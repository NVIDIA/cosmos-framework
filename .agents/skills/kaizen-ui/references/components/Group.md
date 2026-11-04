# Group

A container that joins related items into a single visual unit. Reach for it when adjacent elements should read as one cohesive cluster, such as chips, badges, or button rows.

## Notes

- Use ButtonGroup instead of Group for sets of related action buttons — it ships the correct spacing and button styling per kind.

- Use Stack when sibling items should retain visible spacing; Group removes the gap so items read as one connected element.

## Examples

### Basic Group

Use a flush group to visually join sibling items without any spacing or dividers between them.

```tsx
<Group>
  <Badge>Item A</Badge>
  <Badge>Item B</Badge>
  <Badge>Item C</Badge>
</Group>
```

### Gap Group

Use a gap group when joined items need visible spacing between them while remaining a logical unit.

```tsx
<Group kind="gap">
  <Badge>Item A</Badge>
  <Badge>Item B</Badge>
  <Badge>Item C</Badge>
</Group>
```

### Border Group

Use a border group when items lack their own borders and need a visible separator to distinguish each entry.

```tsx
<Group kind="border">
  <Badge>Item A</Badge>
  <Badge>Item B</Badge>
  <Badge>Item C</Badge>
</Group>
```

### With Badges Group

Use flush with components that already have borders (Badges, Tags, Buttons) to conjoin them into a single visual unit with a 1px overlap.

```tsx
<Group kind="flush">
  <Badge kind="outline">Item 1</Badge>
  <Badge kind="outline">Item 2</Badge>
  <Badge kind="outline">Item 3</Badge>
</Group>
```

### With Tags Group

Use Group to lay out a removable filter chip as a label tag joined to a dismiss tag so each has its own click target while reading as one unit.

```tsx
<Group kind="flush">
  <Tag onClick={handleTagClick}>Filter Name</Tag>
  <Tag aria-label="Remove filter" onClick={handleTagRemove}>
    <Close />
  </Tag>
</Group>
```

### With Input And Button Group

Use to attach an action trigger (dropdown, generate, refresh) directly to a TextInput so the pair reads as one input control.

```tsx
<Group kind="flush">
  <TextInput placeholder="Enter value" />
  <Button aria-label="Open options" kind="secondary">
    <ChevronDown />
  </Button>
</Group>
```

### Single Child Group

Border radius is preserved with a single child, so groups can wrap collections of unknown length (e.g. server-fetched lists) without special-casing the empty / one-item case.

```tsx
<Group kind="flush">
  <Badge kind="outline">Single Item</Badge>
</Group>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| kind | `"gap" \| "flush" \| "border"` | `"flush"` | The visual treatment used to separate adjacent items. - "flush" - Items sit edge-to-edge with no separation, reading as a single conjoined unit. - "gap" - Inserts consistent whitespace between items so they remain distinct but related. - "border" - Draws a divider between items; reach for this when items lack their own border or clear ending. |

## Exports

- Components: `Group`
- Types: `GroupProps`

## See also

- [ButtonGroup](./ButtonGroup.md)
