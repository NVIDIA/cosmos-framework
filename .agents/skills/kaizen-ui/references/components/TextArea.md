# TextArea

A multi-line text input for free-form responses such as descriptions, comments, or notes.

## Notes

- Reach for TextInput when the response will almost always fit on a single line. Use TextArea when multi-line content is expected (descriptions, comments, messages).

- Prefer `resizeable="auto"` so the field grows with content instead of forcing the user to scroll inside a fixed box. Reach for `resizeable="manual"` only when the layout requires a fixed-height field with a drag handle.

- Omitting `resizeable` produces a fixed-height textarea that scrolls internally — this is rarely the right choice; `auto` is almost always preferred for usability.

- Placeholder text is not a label substitute. Wrap in FormField with `slotLabel` or set `aria-label` for standalone fields.

## Examples

### Basic Text Area

```tsx
<TextArea aria-label="Description" defaultValue="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua." />
```

### Controlled Text Area

Controlled mode lets you manage value externally, useful for syncing with form libraries, character counters, or external validation.

```tsx
() => {
  const [value, setValue] = useState("");
  return <TextArea value={value} onValueChange={setValue} placeholder="Type something" />;
};
```

### With Placeholder Text Area

Use placeholder to hint at the expected format. Do not use it as a substitute for a label or required helper text.

```tsx
<TextArea placeholder="Enter a description..." />
```

### Auto Resize Text Area

Use when the textarea should grow with content up to a max height, avoiding scrollbars for short entries. Override `--max-auto-height` (default 400px) to change the cap.

```tsx
<TextArea resizeable="auto" placeholder="Start typing..." />
```

### Manual Resize Text Area

Use when the user should control the textarea height manually via a drag handle in the bottom-right corner.

```tsx
<TextArea resizeable="manual" placeholder="Drag to resize" />
```

### Size Text Area

Default to `medium`. Reserve `small` for dense-table or toolbar contexts where vertical space is constrained — avoid it in regular forms. Use `large` only when surrounding controls already step up to a larger scale.

```tsx
<Flex direction="col" gap="density-md">
  <TextArea size="small" placeholder="Small" />
  <TextArea size="medium" placeholder="Medium" />
  <TextArea size="large" placeholder="Large" />
</Flex>
```

### Status Text Area

Set `status` to drive validation styling manually — `error` for failed validation (pair with helper text on the wrapping FormField) or `success` to confirm a passing value. Reach for `withValidation` when native HTML constraints can drive the state automatically.

```tsx
<Flex direction="col" gap="density-md">
  <TextArea aria-label="Invalid description" status="error" defaultValue="Invalid content" />
  <TextArea aria-label="Valid description" status="success" defaultValue="Looks good" />
</Flex>
```

### With Validation Text Area

Set `withValidation` to drive status styling from the browser's native `:user-valid` / `:user-invalid` pseudo-classes — no JS state required. Combine with `required`, `minLength`, `maxLength`, etc.

```tsx
<TextArea withValidation required placeholder="Will turn red on blur if empty" />
```

### Disabled Text Area

Use when the field is unavailable due to external state (e.g. permissions). Prefer `readOnly` when the value should still be selectable.

```tsx
<TextArea aria-label="Disabled description" disabled defaultValue="Read-only content" />
```

### Read Only Text Area

Use when the value should remain visible and copyable but cannot be edited — common in review or summary screens.

```tsx
<TextArea aria-label="Read-only description" readOnly defaultValue="Read-only value" />
```

### With Slots Text Area

Use `slotStart` / `slotEnd` to attach affordances like icons or action buttons. Use `mb-auto` / `mt-auto` utilities to anchor slot content to the top or bottom of a multi-line field.

```tsx
<TextArea
  placeholder="Add a comment..."
  slotStart={<Document className="mb-auto" />}
  slotEnd={
    <Button className="mt-auto" color="brand" size="small" aria-label="Send">
      <ChevronRight />
    </Button>
  }
/>
```

### Vertical Layout Text Area

Use `layout="vertical"` to stack slot content above and below the textarea — ideal for chat composers, comment boxes, or toolbars that need full-width controls.

```tsx
<TextArea
  layout="vertical"
  placeholder="Compose your message..."
  slotStart={
    <Badge color="green" kind="solid">
      Active
    </Badge>
  }
  slotEnd={
    <Flex gap="2" className="w-full">
      <Button kind="tertiary" size="small" aria-label="Attach">
        <Document />
      </Button>
      <Button className="ml-auto" color="brand" size="small" aria-label="Send">
        <ChevronRight />
      </Button>
    </Flex>
  }
/>
```

### With Form Field Text Area

Wrap TextArea in FormField to attach a label, helper text, and validation status that stay in sync with the field via context.

```tsx
<FormField slotLabel="Description" slotHelp="Your description will be used to generate a report.">
  <TextArea placeholder="Enter your description" required />
</FormField>
```

### Composed

Use the composed primitives when you need full control over the shell and textarea element layout — for example, to mix the textarea with sibling buttons inside the same shell.

```tsx
<InputShell>
  <TextAreaElement aria-label="Text area" defaultValue="Composed primitives" resizeable="auto" />
</InputShell>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ TextAreaElement?: TextAreaElementProps }` | - | Native HTML attributes forwarded to the internal composed components. |
| disableFocusRedirect | `boolean` | `false` | When true, the input will not redirect focus to the input when clicked. |
| kind | `"flat" \| "floating"` | `"flat"` | Visual treatment of the shell. `"flat"` has a border and background; `"floating"` is borderless and transparent. |
| layout | `"horizontal" \| "vertical"` | `"horizontal"` | Axis along which slotted content is arranged inside the shell. `"vertical"` stacks slots in a column with auto height block padding. |
| onValueChange | `(value: string, event: React.ChangeEvent<HTMLTextAreaElement, Element>) => void` | - | Called with the new string value when the textarea content changes. Pair with `value` for controlled state. |
| resizeable | `"auto" \| "manual"` | - | Controls how the textarea can be resized. - "manual" - The user can drag a handle to resize vertically. - "auto" - Grows to fit content up to `--max-auto-height` (defaults to 400px). |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Overall height and typography of the shell. |
| slotEnd | `ReactNode` | - | Content rendered after the textarea. Moves below the textarea when `layout="vertical"`. |
| slotStart | `ReactNode` | - | Content rendered before the textarea. Moves above the textarea when `layout="vertical"`. |
| status | `"error" \| "success"` | - | The status of the input. Use `withValidation` to automatically apply success/error states based on `:user-valid` and `:user-invalid` pseudo classes. |
| value | `string` | - | Controlled textarea value. Pair with `onChange` or `onValueChange` to keep it editable. |
| withValidation | `boolean` | - | Surfaces success and error styling automatically based on the inner input's `:user-valid` and `:user-invalid` states. |

## Exports

- Components: `TextArea`, `TextAreaElement`
- Types: `TextAreaElementProps`, `TextAreaProps`

## See also

- [TextInput](./TextInput.md)
- [FormField](./FormField.md)
