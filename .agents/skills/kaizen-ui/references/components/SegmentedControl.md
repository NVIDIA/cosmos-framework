# SegmentedControl

A horizontal group of mutually exclusive options where selection takes effect immediately. Use for switching between alternate views or modes of the same content.

## Notes

- Use SegmentedControl to switch views (grid vs list, day vs week). For form input where the choice is submitted with other fields, use RadioGroup instead.

- Use Tabs when each option reveals different content. Users expect view-switching from a SegmentedControl, so swapping in different content confuses the interaction model.

- Use ButtonGroup for sets of related but independent actions; SegmentedControl is for mutually exclusive selections.

- Cap at 2-4 items. More options make the control too wide and harder to scan.

- Use `size="small"` only when an explicit dense-layout requirement calls for it, prefer default otherwise.

## Examples

### Basic Segmented Control

```tsx
<SegmentedControl items={["Apple", "Banana", "Cherry"]} />
```

### Default Value Segmented Control

Use when a specific option should be pre-selected on initial render.

```tsx
<SegmentedControl items={["Apple", "Banana", "Cherry"]} defaultValue="Banana" />
```

### Object Items Segmented Control

Use object items when the display label differs from the underlying value, or when you need to forward extra props (like `aria-label` or `disabled`) to individual radio inputs.

```tsx
<SegmentedControl
  items={[
    { value: "grid", children: "Grid" },
    { value: "list", children: "List" },
    { value: "table", children: "Table" },
  ]}
/>
```

### Size Segmented Control

Default to `medium` to match NVIDIA's uniform control-height rule. Reach for `tiny` or `small` only when an explicit dense-layout requirement calls for it, and `large` for prominent actions.

```tsx
<Flex gap="2" direction="col" align="start">
  <SegmentedControl items={["Day", "Week", "Month"]} size="tiny" />
  <SegmentedControl items={["Day", "Week", "Month"]} size="small" />
  <SegmentedControl items={["Day", "Week", "Month"]} size="medium" />
  <SegmentedControl items={["Day", "Week", "Month"]} size="large" />
</Flex>
```

### With Icons Segmented Control

Use icon-only options for compact toolbars. Always supply `aria-label` on each item so screen-reader users get an accessible name.

```tsx
<SegmentedControl
  defaultValue="list"
  items={[
    {
      value: "list",
      "aria-label": "List view",
      children: <Menu />,
    },
    {
      value: "calendar",
      "aria-label": "Calendar view",
      children: <Calendar />,
    },
    {
      value: "docs",
      "aria-label": "Document view",
      children: <Document />,
    },
  ]}
/>
```

### Full Width Segmented Control

Use when the control should stretch to fill its container, such as in a toolbar or settings panel.

```tsx
<SegmentedControl className="w-full" items={["Day", "Week", "Month"]} />
```

### In Form Segmented Control

SegmentedControl renders a native radio group, so the selected option is submitted with the surrounding form under the `name` prop — no extra glue code required.

```tsx
<form onSubmit={handleSubmit}>
  <SegmentedControl name="view" items={["Grid", "List", "Table"]} />
</form>
```

### Controlled Segmented Control

Reach for controlled mode when selection state lives outside the component — e.g. when syncing with URL params, a form library, or driving other UI from the active segment.

```tsx
() => {
  const [selected, setSelected] = useState("Banana");
  return <SegmentedControl items={["Apple", "Banana", "Cherry"]} value={selected} onValueChange={setSelected} />;
};
```

### Composed

Use the composed primitives when you need full control over individual items, such as wrapping each input in custom layout or interleaving non-radio content.

```tsx
<SegmentedControlRoot>
  <SegmentedControlItem>
    <SegmentedControlInput defaultChecked value="grid" />
    Grid
  </SegmentedControlItem>
  <SegmentedControlItem>
    <SegmentedControlInput value="list" />
    List
  </SegmentedControlItem>
</SegmentedControlRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `(string \| { value: string } & Omit<React.ComponentPropsWithRef<"input">, "autocomplete" \| "value">)[]` | - | Options rendered as segments. Pass a string to use it as both the value and label, or an object for full control (e.g. `aria-label` on icon-only segments). |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| defaultValue | `string` | - | Initial selected value when uncontrolled. Defaults to the first item. |
| name | `string` | - | Shared HTML `name` for the inner radio inputs. Falls back to a generated id when omitted. |
| onValueChange | `(value: string) => void` | - | Called when the selected segment changes. |
| size | `"small" \| "tiny" \| "medium" \| "large"` | `"medium"` | Size of the segmented control. |
| value | `string` | - | Controlled selected value. When provided, the consumer is responsible for updating it in response to `onValueChange`. |

`* = required prop`

## Exports

- Components: `SegmentedControl`, `SegmentedControlInput`, `SegmentedControlItem`, `SegmentedControlRoot`
- Types: `SegmentedControlInputProps`, `SegmentedControlItemProps`, `SegmentedControlProps`, `SegmentedControlRootProps`

## See also

- [Tabs](./Tabs.md)
- [RadioGroup](./RadioGroup.md)
- [ButtonGroup](./ButtonGroup.md)
