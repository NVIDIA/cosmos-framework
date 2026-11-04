# NumberInput

A TextInput preset to `type="number"` for numeric-only entry.

## Notes

- Prefer NumberInput over `<TextInput type="number" />` for numeric fields.

- Use the native `min`, `max`, and `step` attributes to constrain valid values.

## Examples

### Basic Number Input

```tsx
<NumberInput aria-label="Quantity" placeholder="0" />
```

### With Form Field Number Input

Wrap in FormField to attach a label and helper text. FormField automatically renders the required indicator when the wrapped input has `required` set. Use the native `min`, `max`, and `step` attributes to constrain valid values.

```tsx
<FormField name="quantity" slotLabel="Quantity" slotHelp="Enter a value between 0 and 10.">
  <NumberInput min={0} max={10} step={1} required />
</FormField>
```

### States Number Input

Use disabled when the field is locked by external conditions and cannot be changed. Use readOnly when the value should remain visible, focusable, and copyable but not editable.

```tsx
<Stack gap="density-md">
  <NumberInput aria-label="Disabled quantity" disabled defaultValue="10" />
  <NumberInput aria-label="Read-only quantity" readOnly defaultValue="10" />
</Stack>
```

### Controlled Number Input

Use controlled mode when state lives outside the input — e.g. when syncing with form libraries or URL state.

```tsx
() => {
  const [value, setValue] = useState("");
  return <NumberInput aria-label="Quantity" value={value} onValueChange={setValue} placeholder="0" />;
};
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ Input?: ComponentPropsWithRef<"input">; InputDismissButton?: ComponentProps<typeof InputDismissButton> }` | - | Native HTML attributes forwarded to the internal composed components. |
| disableFocusRedirect | `boolean` | `false` | When true, the input will not redirect focus to the input when clicked. |
| dismissible | `boolean` | ``true` when `type="search"`, otherwise `false`` | Renders a dismiss button that clears the input and returns focus to it. |
| kind | `"flat" \| "floating"` | `"flat"` | Visual treatment of the shell. `"flat"` has a border and background; `"floating"` is borderless and transparent. |
| layout | `"horizontal" \| "vertical"` | `"horizontal"` | Axis along which slotted content is arranged inside the shell. `"vertical"` stacks slots in a column with auto height block padding. |
| onDismiss | `MouseEventHandler<HTMLButtonElement>` | - | Overrides the default behavior of the dismiss button. |
| onValueChange | `(value: string, event: React.ChangeEvent<HTMLInputElement, Element>) => void` | - | Called with the new string value when the input content changes. Pair with `value` for controlled state. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Overall height and typography of the shell. |
| slotEnd | `ReactNode` | - | Content rendered after the input. Moves below the input when `layout="vertical"`. |
| slotStart | `ReactNode` | - | Content rendered before the input. Moves above the input when `layout="vertical"`. For `type="time"`, `type="date"`, and `type="datetime-local"`, a picker button is always rendered first and this content renders after it. |
| status | `"error" \| "success"` | - | The status of the input. Use `withValidation` to automatically apply success/error states based on `:user-valid` and `:user-invalid` pseudo classes. |
| value | `string` | - | Controlled input value. Pair with `onChange` or `onValueChange` to keep it editable. |
| withValidation | `boolean` | - | Surfaces success and error styling automatically based on the inner input's `:user-valid` and `:user-invalid` states. |

## Exports

- Components: `NumberInput`
- Types: `NumberInputProps`

## See also

- [TextInput](./TextInput.md)
