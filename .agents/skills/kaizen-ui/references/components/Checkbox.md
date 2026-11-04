# Checkbox

A form control that lets users toggle a single option on or off. Supports a mixed state for representing partially-selected groups of related options.

## Notes

- Use Checkbox for values submitted with a form ("apply on submit") and for terms or consent acknowledgments. For binary settings that take effect immediately, use Switch.

- Mental model: Switch communicates "this takes effect immediately"; Checkbox communicates "this will be submitted with the form".

- Use a group of Checkboxes for multiple independent selections. For mutually exclusive choices, use RadioGroup; for 5+ options, use Select or Combobox with `multiple`.

- Always provide `slotLabel`, or `aria-label` when surrounding context already names the option, unless labeling is already provided(FormField, wrapped in <label>, etc.)

- Use `checked="indeterminate"` for parent checkboxes representing a partially-selected group (e.g. "Select all" when some but not all children are checked). Indeterminate is a visual state only — it does not affect form submission.

- When a "Select all" checkbox is present, render it above the individual options so users see it before scanning the list.

- Order options by one of: "Most common first" (when usage data supports it), "Logical grouping" (natural categories), or "Alphabetical" (long lists with no hierarchy).

- Common mistakes: using Checkbox for mutually exclusive options (use RadioGroup), a Checkbox group for 7+ options (consider multi Select or Combobox), a single Checkbox for immediate-effect toggles (use Switch).

## Examples

### Basic Checkbox

```tsx
<Flex direction="col" gap="3">
  <Checkbox slotLabel="Item" />
  <Checkbox checked="indeterminate" slotLabel="Item" />
  <Checkbox checked slotLabel="Item" />
  <Checkbox checked error slotLabel="Item" />
</Flex>
```

### With Label Checkbox

The label is automatically associated with the input.

```tsx
<Checkbox slotLabel="Accept terms and conditions" />
```

### Without Label Checkbox

Use aria-label to provide a label for the checkbox when no label is provided via `slotLabel` or associating via `htmlFor`.

```tsx
<Checkbox aria-label="Accept terms and conditions" />
```

### With Callback Checkbox

```tsx
<Checkbox onCheckedChange={(checked) => console.log(checked)} slotLabel="Toggle me" />
```

### Indeterminate Checkbox

Use `checked="indeterminate"` on a parent checkbox to represent a partially-selected group. The state is visual only and is not submitted with the form.

```tsx
<Checkbox checked="indeterminate" slotLabel="Select all" />
```

### Disabled Checkbox

Should generally be paired with an accompanying tooltip or description to explain why it's disabled.

```tsx
<Checkbox disabled slotLabel="Unavailable option" />
```

### Error Checkbox

Renders in an error state.

```tsx
<Checkbox error slotLabel="Required field" />
```

### Label Left Checkbox

```tsx
<Checkbox labelSide="left" slotLabel="Left-side label" />
```

### Composed

```tsx
<CheckboxRoot>
  <CheckboxInput defaultChecked id="my-checkbox" />
  <Label htmlFor="my-checkbox">Checkbox Label</Label>
</CheckboxRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ CheckboxInput?: NativeElementAttributes<"input", typeof CheckboxInput>; Label?: NativeElementAttributes<"label", typeof Label> }` | - | Native HTML attributes forwarded to the internal composed components. |
| checked | `"indeterminate" \| boolean` | - | Controlled checked state; pair with `onCheckedChange` to handle updates. |
| defaultChecked | `"indeterminate" \| boolean` | - | Initial checked state when the checkbox is uncontrolled, including the tri-state `"indeterminate"` value. |
| disabled | `boolean` | - | Disables interaction with the checkbox. |
| error | `boolean` | - | Renders the checkbox in an error state. |
| form | `string` | - | ID of the form element to associate with, allowing the checkbox to be rendered outside that form. |
| labelSide | `"right" \| "left"` | `"right"` | Side of the checkbox the label is rendered on. |
| name | `string` | - | Form field name submitted with the checkbox value as a name/value pair. |
| onCheckedChange | `(checked: "indeterminate" \| boolean) => void` | - | Called when the checked state changes. |
| required | `boolean` | - | Requires the checkbox to be checked for form submission. |
| slotLabel | `ReactNode` | - | Label rendered next to the checkbox and automatically associated with it for clicks and assistive tech. |

## Exports

- Components: `Checkbox`, `CheckboxInput`, `CheckboxRoot`
- Hooks: `useIndeterminateInputRef`
- Types: `CheckboxInputProps`, `CheckboxProps`, `CheckboxRootProps`, `CheckedState`

## See also

- [Switch](./Switch.md)
- [RadioGroup](./RadioGroup.md)
- [Select](./Select.md)
- [FormField](./FormField.md)
