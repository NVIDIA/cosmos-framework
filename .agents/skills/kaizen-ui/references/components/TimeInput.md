# TimeInput

A TextInput preset to `type="time"` for time-of-day entry, including the built-in native time-picker button.

## Notes

- Prefer TimeInput over `<TextInput type="time" />` for time-of-day fields.

- Set `step={60}` to hide seconds in the native picker.

## Examples

### Basic Time Input

```tsx
<TimeInput aria-label="Time" />
```

### With Form Field Time Input

Wrap in FormField to attach a label. FormField automatically renders the required indicator when the wrapped input has `required` set. Set `step={60}` to hide seconds in the native picker.

```tsx
<FormField name="start-time" slotLabel="Start time">
  <TimeInput step={60} required />
</FormField>
```

### States Time Input

Use disabled when the field is locked by external conditions and cannot be changed. Use readOnly when the value should remain visible, focusable, and copyable but not editable.

```tsx
<Stack gap="density-md">
  <TimeInput aria-label="Disabled time" disabled defaultValue="09:00" />
  <TimeInput aria-label="Read-only time" readOnly defaultValue="09:00" />
</Stack>
```

### Controlled Time Input

Use controlled mode when state lives outside the input — e.g. when syncing with form libraries or URL state.

```tsx
() => {
  const [value, setValue] = useState("09:00");
  return <TimeInput aria-label="Time" value={value} onValueChange={setValue} />;
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

- Components: `TimeInput`
- Types: `TimeInputProps`

## See also

- [TextInput](./TextInput.md)
