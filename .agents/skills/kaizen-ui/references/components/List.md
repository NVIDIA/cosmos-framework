# List

Groups related items into a sequence, supporting numbered, bulleted, and icon-marked formats.

## Notes

- Reach for List when content is semantically a list (steps, features, bullet points) so the rendered `<ul>` / `<ol>` carries the right semantics. Use Stack for layout-only sequences and Menu for selectable item lists.

- Use `kind="ordered"` only when item sequence carries meaning (instructions, rankings). Default to `"unordered"`.

- Keep `slotMarker` icons consistent across every item in the same list — mixing markers makes the list read as ad-hoc rows rather than a list.

## Examples

### Basic List

```tsx
<List items={["Design tokens", "Components", "Patterns"]} />
```

### Ordered List

Use when items follow a specific sequence such as instructions or steps.

```tsx
<List kind="ordered" items={["Install the package", "Import the component", "Render it"]} />
```

### With Custom Marker List

Override per-item markers with a JSX node (icon) or a string. Items left as plain strings keep the default marker for the list kind.

```tsx
<List
  items={[
    "Design tokens",
    {
      children: "Components",
      slotMarker: <ChevronRight />,
    },
    { children: "Patterns", slotMarker: "•" },
  ]}
/>
```

### Composed

```tsx
<ListRoot kind="unordered">
  <ListItem>
    <ListItemMarker />
    Design tokens
  </ListItem>
  <ListItem>
    <ListItemMarker />
    Components
  </ListItem>
  <ListItem>
    <ListItemMarker />
    Patterns
  </ListItem>
</ListRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `(string \| { children: ReactNode; slotMarker?: ReactNode })[]` | - | Items to render in order. Pass strings for plain text or objects for custom content and markers. |
| attributes | `{ ListItem?: NativeElementAttributes<"li", typeof ListItem>; ListItemMarker?: NativeElementAttributes<"div", typeof ListItemMarker> }` | - | Native HTML attributes forwarded to the internal composed components. |
| kind | `"ordered" \| "unordered"` | `"unordered"` | List style. `"ordered"` numbers items in sequence; `"unordered"` renders them as peers with bullets. |

`* = required prop`

## Exports

- Components: `List`, `ListItem`, `ListItemMarker`, `ListRoot`
- Types: `ListItemData`, `ListItemMarkerProps`, `ListItemProps`, `ListProps`, `ListRootOrderedProps`, `ListRootProps`, `ListRootUnorderedProps`
