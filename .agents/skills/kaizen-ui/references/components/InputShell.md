# InputShell

A low-level wrapper that styles any element as a KUI input. Reach for the higher-level inputs first; use this only when composing a custom control.

## Notes

- Prefer TextInput, TextArea, Select, or Combobox before reaching for InputShell. Use InputShell only when wrapping a non-standard input that the higher-level components don't cover, or for a custom input.

- Choose by scenario: standard text input → TextInput; standard dropdown → Select; standard multi-line → TextArea; searchable dropdown → Combobox; custom input (color picker, token input, rich editor, etc.) → InputShell wrapping the custom element.

- The wrapped element must be an `input`, `textarea`, `select`, or a `button[data-input-slot]` — focus redirect and styling hooks expect one of these targets.

## Examples

### Basic Input Shell

```tsx
<InputShell>
  <input type="text" placeholder="Enter text" />
</InputShell>
```

### With Icons Input Shell

Slot icons or other content directly before/after the input — children are laid out in a row by default.

```tsx
<InputShell>
  <Bell />
  <input type="text" placeholder="Search..." />
  <Close />
</InputShell>
```

### Textarea Input Shell

Wrap a native textarea to get the same shell styling and focus redirect as text inputs.

```tsx
<InputShell>
  <textarea placeholder="Enter a longer message" />
</InputShell>
```

### Select Input Shell

Wrap a native select. Use a disabled hidden empty option with `required` to show placeholder text.

```tsx
<InputShell>
  <select aria-label="Example option" required defaultValue="" name="example-select">
    <option value="" disabled hidden>
      Select an option
    </option>
    <option value="1">Option 1</option>
    <option value="2">Option 2</option>
    <option value="3">Option 3</option>
  </select>
</InputShell>
```

### Button As Input Shell

Add `data-input-slot` to a button to style triggers (popovers, modals, custom pickers) like a form field. Use `data-has-selected-value="false"` to apply placeholder styling.

```tsx
<InputShell>
  <button data-input-slot data-has-selected-value="false" type="button">
    Select a value...
  </button>
</InputShell>
```

### Floating Input Shell

Use the floating kind for borderless, transparent inputs in toolbars, inline editing, or embedded contexts.

```tsx
<InputShell kind="floating">
  <input type="text" placeholder="Floating input" />
</InputShell>
```

### Sizes Input Shell

Match the shell to its surroundings: `small` for table cells or dense forms, `medium` (default) for typical form fields, and `large` for hero search bars.

```tsx
<div className="flex flex-col gap-2">
  <InputShell size="small">
    <input type="text" placeholder="Small" />
  </InputShell>
  <InputShell size="medium">
    <input type="text" placeholder="Medium" />
  </InputShell>
  <InputShell size="large">
    <input type="text" placeholder="Large" />
  </InputShell>
</div>
```

### Vertical Layout Input Shell

Use vertical layout when slotted content needs to stack above or below the input rather than sit beside it.

```tsx
<InputShell layout="vertical">
  <span>Label above</span>
  <input type="text" placeholder="Vertical layout" />
</InputShell>
```

### Status Input Shell

Set `data-status="success"` or `data-status="error"` on the inner input for explicit validation feedback. Use `withValidation` instead when you want native `:user-valid` / `:user-invalid` to drive the styling.

```tsx
<InputShell>
  <input aria-label="Invalid value" data-status="error" defaultValue="invalid-value" />
</InputShell>
```

### With Validation Input Shell

Use when native HTML constraints (`required`, `pattern`, etc.) should drive success/error styling automatically via `:user-valid` and `:user-invalid`.

```tsx
<InputShell withValidation>
  <input type="email" placeholder="email@example.com" required />
</InputShell>
```

### Disabled Input Shell

The shell picks up disabled styling automatically when the inner element has the `disabled` attribute. Use `data-disabled="true"` on the shell itself for purely visual disabled state.

```tsx
<InputShell>
  <input aria-label="Disabled input" disabled defaultValue="Disabled input" />
</InputShell>
```

### Read Only Input Shell

For read-only data display. Only `input` and `textarea` support `readOnly` — for select-based fields, render an `input` with the resolved value instead.

```tsx
<InputShell>
  <input aria-label="Read-only input" readOnly defaultValue="Read-only value" />
</InputShell>
```

### Disable Focus Redirect Input Shell

Disable the click-to-focus behavior when the shell contains its own interactive elements (buttons, links) that should receive their own clicks.

```tsx
<InputShell disableFocusRedirect>
  <input type="text" placeholder="Click shell — focus stays put" />
</InputShell>
```

### As Child Input Shell

Render the shell as a custom element (typically a `label` wrapping a file input). Pair with `disableFocusRedirect` so the label — not the shell — handles activating the input.

```tsx
<InputShell asChild disableFocusRedirect>
  <label>
    <input aria-label="Upload file" type="file" />
  </label>
</InputShell>
```

### Composed

Compose `InputDismissButton` (and other slot content like icons) inside `InputShell` for clearable inputs. The dismiss button auto-hides when the input is empty; wire up `onClick` to clear the value yourself.

```tsx
<InputShell>
  <Bell />
  <input aria-label="Users input" type="text" defaultValue="users matching 'admin'" />
  <InputDismissButton />
</InputShell>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| disableFocusRedirect | `boolean` | `false` | When true, the input will not redirect focus to the input when clicked. |
| kind | `"flat" \| "floating"` | `"flat"` | Visual treatment of the shell. `"flat"` has a border and background; `"floating"` is borderless and transparent. |
| layout | `"horizontal" \| "vertical"` | `"horizontal"` | Axis along which slotted content is arranged inside the shell. `"vertical"` stacks slots in a column with auto height block padding. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Overall height and typography of the shell. |
| withValidation | `boolean` | - | Surfaces success and error styling automatically based on the inner input's `:user-valid` and `:user-invalid` states. |

## Exports

- Components: `InputDismissButton`, `InputShell`
- Types: `InputShellProps`, `WithInputShellStatus`
