# ToggleButtonGroup

A set of related toggle buttons that each switch an option on or off, with one tab stop for the whole set. Use for options that apply to the same content, such as text formatting or which panels are visible.

## Notes

- The group needs an accessible name: pass `aria-label` ("Text formatting") or `aria-labelledby` pointing at a visible heading.

- Keep the default `selectionMode="multiple"` when options combine (Bold + Italic). Use `selectionMode="single"` only when at most one option can be on and having none on is valid (a status filter where no selection shows everything). When exactly one option must always be selected, use SegmentedControl instead; for a choice submitted with a form, use RadioGroup or a group of Checkboxes, because the group submits no form value.

- ButtonGroup is for related actions that do not leave a state behind (Undo, Redo). Use ToggleButtonGroup only when each button shows a persistent on/off state.

- Pass each item's icon to its `slotIcon` so it switches from line to fill while the item is pressed. Icon-only items omit `children` and must set `aria-label`. Limit a group to the options that act on the same content; split unrelated options into separate groups.

- Set `orientation="vertical"` when the group sits in a vertical toolbar or side panel, so Up and Down move focus between items.

## Examples

### Basic Toggle Button Group

Pass each item's icon to `slotIcon` so it fills while the item is pressed. Icon-only items need an `aria-label`.

```tsx
<ToggleButtonGroup
  aria-label="Text formatting"
  defaultValue={["bold"]}
  items={[
    { value: "bold", "aria-label": "Bold", slotIcon: <Bold /> },
    { value: "italic", "aria-label": "Italic", slotIcon: <Italic /> },
    {
      value: "underline",
      "aria-label": "Underline",
      slotIcon: <Underline />,
    },
  ]}
/>
```

### Single Select Toggle Button Group

Use `selectionMode="single"` when at most one option can be on and none is a valid state, such as a status filter where no selection shows everything. Pressing the pressed item releases it. When one option must always be selected, use SegmentedControl instead.

```tsx
<ToggleButtonGroup
  aria-label="Filter by status"
  selectionMode="single"
  items={[
    { value: "running", children: "Running" },
    { value: "queued", children: "Queued" },
    { value: "failed", children: "Failed" },
  ]}
/>
```

### Controlled Toggle Button Group

Control `value` when the pressed options drive other UI, such as the series a chart plots, or live in URL state. `onValueChange` receives the values of every pressed item.

```tsx
() => {
  const [series, setSeries] = useState<string[]>(["gpu-util", "memory"]);
  return (
    <ToggleButtonGroup
      aria-label="Chart series"
      value={series}
      onValueChange={setSeries}
      items={[
        { value: "gpu-util", children: "GPU util" },
        { value: "memory", children: "Memory" },
        { value: "power", children: "Power" },
      ]}
    />
  );
};
```

### Composed

Compose the group from items when an item needs a wrapper the `items` array cannot express, such as a Tooltip that shows the keyboard shortcut for an icon-only item.

```tsx
<ToggleButtonGroupRoot aria-label="Text formatting">
  <Tooltip slotContent="Bold (Ctrl+B)">
    <ToggleButtonGroupItem aria-label="Bold" slotIcon={<Bold />} value="bold" />
  </Tooltip>
  <Tooltip slotContent="Italic (Ctrl+I)">
    <ToggleButtonGroupItem aria-label="Italic" slotIcon={<Italic />} value="italic" />
  </Tooltip>
  <Tooltip slotContent="Underline (Ctrl+U)">
    <ToggleButtonGroupItem aria-label="Underline" slotIcon={<Underline />} value="underline" />
  </Tooltip>
</ToggleButtonGroupRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `(Pick<ToggleButtonGroupItemProps, "style" \| "className" \| "aria-label" \| "disabled" \| "value" \| "onPressedChange" \| "slotIcon" \| "slotPressed"> & { children?: React.ReactNode; attributes?: { ToggleButtonGroupItem?: NativeElementAttributes<"button", React.ForwardRefExoticComponent<Omit<ToggleButtonGroupItemProps, "ref"> & React.RefAttributes<HTMLButtonElement>>>; }; })[]` | - | Toggle button definitions rendered in order. |
| color | `"brand" \| "neutral"` | - | Color applied to every item in the group. |
| defaultValue | `string[]` | `[]` | Values of the items pressed on first render. Use when the selection is uncontrolled. |
| disabled | `boolean` | - | Disables every item in the group. An item's own `disabled` takes precedence. |
| loop | `boolean` | `true` | Whether arrow-key navigation wraps from the last item back to the first. |
| onValueChange | `(value: string[]) => void` | - | Called with the values of every pressed item when the selection changes. |
| orientation | `"horizontal" \| "vertical"` | `"horizontal"` | Layout direction. Determines both the visual arrangement and which arrow keys move focus. |
| selectionMode | `"multiple" \| "single"` | `"multiple"` | How many items can be pressed at once. - "multiple" - Each item toggles on its own, such as Bold, Italic, and Underline. - "single" - Pressing an item releases the others, and pressing the pressed item releases it, leaving none pressed. |
| size | `"small" \| "tiny" \| "medium" \| "large"` | - | Size applied to every item in the group. |
| value | `string[]` | - | Controlled values of the pressed items. Pair with `onValueChange` to manage state externally. |

`* = required prop`

## Exports

- Components: `ToggleButtonGroup`, `ToggleButtonGroupItem`, `ToggleButtonGroupRoot`
- Types: `ToggleButtonGroupItemProps`, `ToggleButtonGroupProps`, `ToggleButtonGroupRootProps`

## See also

- [ToggleButton](./ToggleButton.md)
- [SegmentedControl](./SegmentedControl.md)
- [ButtonGroup](./ButtonGroup.md)
