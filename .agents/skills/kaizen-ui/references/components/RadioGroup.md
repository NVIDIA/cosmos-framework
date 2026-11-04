# RadioGroup

A set of mutually exclusive options where the user picks exactly one. Use when the available options can comfortably fit on screen at once.

## Notes

- RadioGroup works best with 2-4 options. For 5+ mutually exclusive options use Select; for switching between views or modes of the same content use SegmentedControl; for multi-select use a group of Checkboxes.

- SegmentedControl is for switching between views of the same content, not for choosing a value to submit. Use RadioGroup whenever the choice is part of a form.

- Default to `orientation="vertical"`. Use `orientation="horizontal"` only when every option label is short (1-2 words) and horizontal compactness benefits the form.

- Use `kind="tile"` when each option needs a larger clickable surface or supporting details. Pair it with `orientation="horizontal"` only when the tiles remain easy to compare at the available width.

- Pre-select the most common option via `defaultValue` when one choice is clearly dominant — users keep the default they would have picked anyway.

- Order options deliberately (most common, logical sequence, alphabetical, or safest first when options have risk implications — non-destructive or reversible option first) and keep the ordering consistent across sessions to build muscle memory.

- Use the per-item `danger` flag for individual options with destructive implications. If the entire choice is dangerous, prefer marking the form's submit button with `color="danger"` instead.

## Examples

### Basic Radio Group

```tsx
<RadioGroup
  name="color"
  defaultValue="blue"
  items={[
    { children: "Red", value: "red" },
    { children: "Blue", value: "blue" },
    { children: "Green", value: "green" },
  ]}
/>
```

### With Icons Radio Group

Use icons in `children` to give each option a visual anchor when the label alone is ambiguous.

```tsx
<RadioGroup
  name="backup-frequency"
  defaultValue="weekly"
  items={[
    {
      children: (
        <>
          <CheckCircle />
          Daily
        </>
      ),
      value: "daily",
    },
    {
      children: (
        <>
          <Clock />
          Weekly
        </>
      ),
      value: "weekly",
    },
    {
      children: (
        <>
          <Calendar />
          Monthly
        </>
      ),
      value: "monthly",
    },
  ]}
/>
```

### Tile Radio Group

Use tile radio groups when each option needs a larger click target, supporting details, or a selected tile affordance. Tiles are density responsive, so apply `nv-density-compact` or `nv-density-spacious` to tune their spacing. By default horizontal tiles will wrap and have a min width of 10rem - override the min width by setting `--tile-min-width` if you need it to be larger/smaller.

```tsx
<RadioGroup
  name="compute-profile"
  kind="tile"
  orientation="horizontal"
  defaultValue="balanced"
  items={[
    {
      children: (
        <Stack gap="1">
          <Text kind="label/semibold/md">Balanced</Text>
          <Text className="text-secondary" kind="body/regular/sm">
            Good default for mixed workloads.
          </Text>
        </Stack>
      ),
      value: "balanced",
    },
    {
      children: (
        <Stack gap="1">
          <Text kind="label/semibold/md">Performance</Text>
          <Text className="text-secondary" kind="body/regular/sm">
            Prioritizes throughput over cost.
          </Text>
        </Stack>
      ),
      value: "performance",
    },
    {
      children: (
        <Stack gap="1">
          <Text kind="label/semibold/md">Economy</Text>
          <Text className="text-secondary" kind="body/regular/sm">
            Conserves resources for lighter tasks.
          </Text>
        </Stack>
      ),
      value: "economy",
    },
  ]}
/>
```

### With Danger Item Radio Group

Use the `danger` flag on an item to warn users about a destructive or irreversible choice.

```tsx
<RadioGroup
  name="action"
  defaultValue="keep"
  items={[
    { children: "Keep data", value: "keep" },
    { children: "Archive data", value: "archive" },
    { children: "Delete permanently", value: "delete", danger: true },
  ]}
/>
```

### Error Radio Group

Use the group-level `error` prop to surface form validation failures across every item — pair with a `FormField` to render the error message.

```tsx
<RadioGroup
  name="payment-method"
  error
  items={[
    { children: "Credit Card", value: "credit-card" },
    { children: "Debit Card", value: "debit-card" },
    { children: "PayPal", value: "paypal" },
  ]}
/>
```

### Label Start Radio Group

Use start label placement when end-aligning radio indicators improves visual scanning.

```tsx
<RadioGroup
  name="align"
  defaultValue="left"
  labelSide="start"
  items={[
    { children: "Left", value: "left" },
    { children: "Center", value: "center" },
    { children: "Right", value: "right" },
  ]}
/>
```

### Hidden Indicator Radio Group

Set `showIndicator={false}` when tile styling should communicate the selected state without a visible radio input.

```tsx
<RadioGroup
  name="plan-tier"
  kind="tile"
  orientation="horizontal"
  defaultValue="pro"
  showIndicator={false}
  items={[
    { children: "Free", value: "free" },
    { children: "Pro", value: "pro" },
    { children: "Enterprise", value: "enterprise" },
  ]}
/>
```

### Composed

Use the composed primitives when you need full control over item layout or custom content around each radio input.

```tsx
<RadioGroupRoot name="gpu" defaultValue="4090" orientation="horizontal">
  <RadioGroupItem>
    <RadioGroupInput value="4090" />
    RTX 4090
  </RadioGroupItem>
  <RadioGroupItem>
    <RadioGroupInput value="4080" />
    RTX 4080
  </RadioGroupItem>
  <RadioGroupItem>
    <RadioGroupInput value="3090" />
    RTX 3090
  </RadioGroupItem>
</RadioGroupRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `(Pick<RadioGroupInputProps, "disabled" \| "value" \| "danger" \| "required"> & Pick<RadioGroupItemProps, "style" \| "className" \| "children"> & { attributes?: { RadioGroupItem?: NativeElementAttributes<"label", React.ForwardRefExoticComponent<Omit<RadioGroupItemProps, "ref"> & React.RefAttributes<HTMLLabelElement>>>; RadioGroupInput?: NativeElementAttributes<"input", React.ForwardRefExoticComponent<Omit<RadioGroupInputProps, "ref"> & React.RefAttributes<HTMLInputElement>>>; }; })[]` | - | Option definitions rendered in order as the radio group's items. |
| defaultValue | `string` | - | Initial selected value when uncontrolled. |
| disabled | `boolean` | - | Disables every radio in the group. |
| error | `boolean` | - | Renders every radio in an error state. |
| kind | `"default" \| "tile"` | `"default"` | Visual treatment for the group and its items. |
| labelSide | `"start" \| "end"` | `"end"` | Which side of the radio input the label renders on. |
| name | `string` | - | Shared HTML `name` applied to each radio input. Falls back to the surrounding `FormField`'s `name` when omitted. |
| onValueChange | `(value: string) => void` | - | Called when the selected value changes. |
| orientation | `"horizontal" \| "vertical"` | `"vertical"` | Layout direction. Determines both visual arrangement and arrow-key navigation order. |
| required | `boolean` | - | Marks the group as required so the surrounding form refuses to submit without a selection. |
| showIndicator | `boolean` | `true` | Shows the radio indicator. When `false`, the input remains in the DOM but is visually hidden so tile-style items can convey the selected state themselves. |
| value | `string` | - | Controlled selected value. When provided, the consumer is responsible for updating it in response to `onValueChange`. |

`* = required prop`

## Exports

- Components: `RadioGroup`, `RadioGroupInput`, `RadioGroupItem`, `RadioGroupRoot`
- Hooks: `useRadioGroupContext`
- Contexts: `RadioGroupContext`
- Types: `RadioGroupContextType`, `RadioGroupInputProps`, `RadioGroupItemProps`, `RadioGroupProps`, `RadioGroupRootProps`
