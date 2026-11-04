# TextInput

A single-line input for text, numbers, or special symbols. Commonly used in forms; switch to TextArea when the response is likely to wrap.

## Notes

- Pick the right primitive before building: TextInput for single-line free text, TextArea when users will regularly need multiple lines, Combobox for a text input paired with a selectable list, InputShell when composing a custom input from scratch. One-line test: if the expected input will almost always fit on a single line, use TextInput.

- When used as a search input, always include a MagnifyingGlass icon via slotStart for affordance.

- For search inputs, set `dismissible={true}` and run client-side filtering on keystroke; do not add a separate "Search" button next to the input unless the search is server-initiated. For form fields where the user edits in place, set `dismissible={false}`.

- Prefer wrapping in FormField (handles label, helper text, and status propagation) and reach for the `status` prop directly only on standalone inputs.

- Set `type` to match the data semantics (`"email"`, `"url"`, `"tel"`, `"password"`, `"search"`) so the browser provides the right keyboard, autofill hints, and validation.

- Prefer DatePicker for date selections

- Placeholder text is not a label substitute. Wrap in FormField with `slotLabel` or set `aria-label` for standalone inputs.

- Avoid placing interactive elements in `slotEnd` — use `dismissible` for clearing or render actions outside the input.

## Examples

### Basic Text Input

```tsx
<Flex direction="col" gap="density-md">
  <TextInput aria-label="Greeting" defaultValue="Hello, world!" />
  <TextInput aria-label="Time" type="time" />
</Flex>
```

### With Slots Text Input

Use slotStart for affordance icons (e.g. a magnifying glass for search) and slotEnd for trailing hints like result counts or unit labels.

```tsx
<TextInput aria-label="Username" slotStart={<Bell />} slotEnd={<span>(7)</span>} defaultValue="admin" />
```

### Search Text Input

Standard search input: `type="search"` and `dismissible` so users can quickly clear the query. Pair with a leading magnifying-glass icon in `slotStart`. Provide `aria-label` when the input is not inside a FormField.

```tsx
<TextInput type="search" aria-label="Search" placeholder="Search..." dismissible slotStart={<SearchIcon />} />
```

### Dismissible Text Input

Use when the user needs a quick way to clear the field, such as search inputs.

```tsx
<TextInput aria-label="Clearable value" dismissible defaultValue="Clearable value" />
```

### Read Only Text Input

Use readOnly (instead of disabled) when the value should remain visible, focusable, and copyable but not editable — common in summary or review screens.

```tsx
<TextInput aria-label="Read-only value" readOnly value="Read-only value" />
```

### Disabled Text Input

Use when the field value is locked by external conditions and cannot be changed.

```tsx
<TextInput aria-label="Disabled value" disabled defaultValue="Cannot edit" />
```

### Size Text Input

Match the input size to surrounding controls — small for dense tables/toolbars, large for hero search bars or onboarding forms.

```tsx
<Stack gap="density-md">
  <TextInput size="small" placeholder="Small" />
  <TextInput size="medium" placeholder="Medium" />
  <TextInput size="large" placeholder="Large" />
</Stack>
```

### Status Text Input

Set `status` to drive validation styling manually — `error` for failed validation or `success` to confirm a passing value. Reach for `withValidation` when native HTML constraints can drive the state automatically.

```tsx
<Stack gap="density-md">
  <TextInput aria-label="Invalid email" status="error" defaultValue="invalid-email" />
  <TextInput aria-label="Valid email" status="success" defaultValue="valid@example.com" />
</Stack>
```

### With Validation Text Input

Use withValidation to drive success/error styling automatically from native HTML constraints (`required`, `pattern`, `type="email"`, etc.) via `:user-valid` and `:user-invalid` — no controlled state needed.

```tsx
<TextInput withValidation required type="email" placeholder="name@example.com" />
```

### Date Text Input

Use type="date" or type="datetime-local" to render a calendar button that opens the browser's native picker via `showPicker()`.

```tsx
<Stack gap="density-md">
  <TextInput aria-label="Date" type="date" />
  <TextInput aria-label="Date and time" type="datetime-local" />
</Stack>
```

### Time Text Input

Use for time selection with a native picker button. Set step={60} to hide seconds.

```tsx
<TextInput aria-label="Time" type="time" step={60} />
```

### With Form Field Text Input

Wrap in FormField to attach a label and helper text. FormField pipes `id`, `name`, and validation status into the input via context, and automatically renders the required indicator when the wrapped input has `required` set.

```tsx
<FormField name="username" slotLabel="Username" slotHelp="3–20 characters, letters and numbers only.">
  <TextInput placeholder="Enter your username" required />
</FormField>
```

### Controlled Text Input

Use controlled mode when state lives outside the input — e.g. when syncing with form libraries or URL state.

```tsx
() => {
  const [value, setValue] = useState("");
  return <TextInput value={value} onValueChange={setValue} placeholder="Type something" />;
};
```

### Debounced Text Input

Combine with `useDebounce` to throttle expensive work like API calls or filtering driven by the input value.

```tsx
() => {
  const [value, setValue] = useState("");
  const debounced = useDebounce(value, 500);
  return (
    <Stack gap="density-md">
      <TextInput value={value} onValueChange={setValue} placeholder="Search..." />
      <Text>Debounced: {debounced}</Text>
    </Stack>
  );
};
```

### Composed

Drop down to InputShell + a raw `<input>` when you need full control over slot content, ARIA, or non-standard layouts that TextInput's API doesn't expose.

```tsx
<InputShell>
  <Bell />
  <input type="text" placeholder="Composed text input" />
</InputShell>
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

- Components: `TextInput`
- Types: `TextInputProps`

## See also

- [FormField](./FormField.md)
- [TextArea](./TextArea.md)
- [InputShell](./InputShell.md)
