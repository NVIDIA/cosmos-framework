# Halo

A decorative rotating gradient ring that wraps a single element to draw attention to it. Halo is presentational only — it adds no semantics, so the element inside keeps its own role, label, and behavior.

## Notes

- Halo wraps exactly one child. Give it a single element — a Button, a Card, a TextInput — not a fragment or a list.

- Halo sizes itself to the child, so it drops into a flex row or grid cell as an auto-sized item. Give it `className="block"` when the child is meant to fill its container instead — a TextInput in a FormField, or a Card that has to match the height of its siblings — otherwise it collapses to the child's own content width.

- Do not set `childRadius`. Halo already matches the ring to Button, the form controls, Badge, Tag, Card, and Panel on its own. Reach for it only when the child rounds to something off the scale, and then pass the scale value that child uses.

- Use Halo on at most one element per view. It is an emphasis treatment, and its impact collapses when several compete on the same screen.

- Halo is purely decorative — never rely on it to communicate state, validation, or selection. Use Badge, StatusIndicator, or StatusMessage for meaning.

## Examples

### Basic Halo

```tsx
<Halo>
  <Button kind="primary">Ask AI</Button>
</Halo>
```

### Halo With Glow

Set `glow` to bloom the gradient outward past the ring. It is off by default because the bloom spills into neighbouring content in dense layouts; reach for it where the element has room around it.

```tsx
<Halo glow>
  <Card>Lorem ipsum dolor sit amet</Card>
</Halo>
```

### Halo That Fills Its Container

Halo sizes itself to its child, so by default it shrinks to the child's content width. Add `className="block"` when the child should fill its container instead — here a search input that has to span the form field.

```tsx
<FormField slotLabel="Search">
  <Halo className="block">
    <TextInput placeholder="Search models, docs, and blueprints" type="search" />
  </Halo>
</FormField>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| childRadius | `"none" \| "density-xl" \| "sm" \| "md" \| "lg" \| "xl" \| "2xl" \| "3xl" \| "round"` | - | Corner rounding of the child, which the ring grows outward from. Halo reads this off the child on its own — leave it unset unless the child rounds to something off the scale, or is a surface Halo does not recognise. |
| glow | `boolean` | `false` | Whether the gradient blooms outward past the ring. The bloom spreads well past the wrapped element, so leave it off in dense layouts. |

## Exports

- Components: `Halo`
- Types: `HaloProps`
