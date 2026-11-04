# AnimatedChevron

A chevron icon that animates between two orientations to indicate the open or closed state of a disclosure. Use inside expand and collapse triggers when users need a visual cue for the current state.

## Notes

- Place AnimatedChevron inside the trigger it describes. Without `state` it follows `data-state="open"` or `data-state="closed"` on an ancestor, which Collapsible and Popover triggers already set, or the open state of the `<details>` whose `<summary>` contains it.

- Pass `state` only when nothing above the chevron sets `data-state`, such as a button that toggles a panel from your own state. `state` takes precedence over any ancestor state.

- The chevron is hidden from assistive technology, so a trigger you build yourself must expose its state with `aria-expanded` and point `aria-controls` at the content it toggles.

- Do not add one to Accordion, Select, Combobox, Dropdown, or DatePicker triggers, which render their own, or to a Collapsible trigger that already sets `showChevron`.

- For an arrow that never changes direction, use a static chevron icon instead.

## Examples

### Basic Usage

Reads open/closed state from `data-state="open"` or `data-state="closed"` from parent.

```tsx
<div data-state="open">
  <AnimatedChevron />
</div>
```

### Controlled Usage

Manually control the state with the `state` prop.

```tsx
<AnimatedChevron state="closed" />
```

### Usage With Collapsible

When used in most of our components, it automatically reads the open/closed state from the parent data attribute, such as Collapsible.

```tsx
<Collapsible
  slotTrigger={
    <Flex gap="density-md" align="center">
      Toggle content <AnimatedChevron />
    </Flex>
  }
>
  ...
</Collapsible>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| state | `"open" \| "closed"` | - | Forces the chevron orientation when the parent element does not set a `data-state` attribute. |

## Exports

- Components: `AnimatedChevron`
- Types: `AnimatedChevronProps`

## See also

- [Collapsible](./Collapsible.md)
