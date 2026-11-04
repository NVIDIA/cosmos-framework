# Toast

A small, temporary overlay that gives the user non-critical feedback after an action. Reach for it when the message is informational and can be missed without consequence.

## Notes

- Toast = transient, low-attention, non-blocking confirmation that the user can miss without consequence. Reach for Notification when the message persists in an inbox until acknowledged, Banner for page-level conditions that should stay visible, and Modal when the user must respond before continuing. StatusMessage is separate because it replaces content rather than overlaying it.

- Toast intentionally biases toward low emphasis — overusing high-attention or persistent messaging causes alert fatigue, which is why transient toasts are the right choice for non-blocking confirmations.

- Do not use Toast for errors, warnings, or any message requiring user action — Toast should auto-dismiss. Use Notification or Banner instead for actionable messages.

- Keep toast text to four words or fewer — long messages get cut off before they can be read. Reach for Notification when more detail is needed.

- Use `slotAction` sparingly — only for "Undo" on destructive actions or "View" to jump to a result. Routine confirmations should not have actions.

- Render `Toaster` once near the app root and call `toast()` to show a toast over the page. Render `Toast` directly only when the message belongs in the layout rather than over it.

- Set `Toaster`'s `position` once for the whole app and keep it there — moving toasts between surfaces makes them easy to miss. Bottom-center suits content-focused apps, top-right matches OS-style notifications, and bottom-right keeps toasts out of primary content.

- Let `Toaster` handle several toasts at once: it caps how many are on screen with `max` and collapses the rest into a stack that expands on hover, touch, or focus. Raising `max` past a handful buries the newest message.

- Prefer a semantic status (`"info"`, `"success"`, `"warning"`, `"error"`) over `status="neutral"`. Reach for `"neutral"` only for custom or non-semantic cases where none of the others fit.

## Examples

### Statuses Toast

Set `status` to convey severity: `info` for neutral confirmation, `success` for completed actions, `warning` to caution about a degraded condition, and `error` for failures that need corrective action.

```tsx
<Flex direction="col" gap="density-md">
  <Toast status="info">Operation completed</Toast>
  <Toast status="success">Changes saved</Toast>
  <Toast status="warning">Connection unstable</Toast>
  <Toast status="error">Upload failed</Toast>
</Flex>
```

### Dismissible Toast

Use when the user should be able to manually dismiss the toast before it auto-hides.

```tsx
<Toast status="info" onClose={handleClose}>
  Session expiring soon
</Toast>
```

### With Action Toast

Use `slotAction` only for low-stakes follow-ups on a successful action ("Undo" after a destructive action, "View" to jump to a result). Do not use Toast for errors or warnings that require user action — reach for Notification or Banner instead.

```tsx
<Toast
  status="success"
  onClose={handleClose}
  slotAction={
    <Button kind="tertiary" size="small">
      Undo
    </Button>
  }
>
  Item deleted
</Toast>
```

### Working Toast

Use to indicate that a background process is actively running.

```tsx
<Toast status="working">Processing request</Toast>
```

### With Custom Icon Toast

Use `slotIcon` to override the default status icon when a domain-specific glyph communicates the message better.

```tsx
<Toast status="success" slotIcon={<Bell />}>
  Reminder scheduled
</Toast>
```

### With Title Toast

Set the native `title` attribute so the full message is available on hover when the toast text is long enough to truncate.

```tsx
<Toast status="info" title="Deployment finished in 3m 24s across 4 regions">
  Deployment finished in 3m 24s across 4 regions
</Toast>
```

### Composed

Use composed primitives when you need full control over the toast layout, icon, and actions.

```tsx
<ToastRoot status="success">
  <ToastContent>
    <ToastIcon status="success" />
    <ToastText>Composed toast message</ToastText>
  </ToastContent>
  <ToastActions>
    <Button kind="tertiary" size="small">
      Undo
    </Button>
  </ToastActions>
</ToastRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ ToastText?: NativeElementAttributes<"span", typeof ToastText>; ToastContent?: NativeElementAttributes<"div", typeof ToastContent>; ToastActions?: NativeElementAttributes<"div", typeof ToastActions>; ToastIcon?: NativeElementAttributes<"div", typeof ToastIcon>; ToastCloseButton?: NativeElementAttributes<"button", typeof Button> }` | - | Native HTML attributes forwarded to the internal composed components. |
| children | `ReactNode` | - | Toast message text. Keep it brief and direct, ideally no more than four words. |
| onClose | `() => void` | - | Called when the close button is clicked. Omit to render the toast without a close button. |
| slotAction | `ReactNode` | - | A single action rendered next to the close button. Typically a `size="small"` button. |
| slotIcon | `ReactNode` | - | Replaces the default status icon. |
| status | `"neutral" \| "error" \| "warning" \| "success" \| "info" \| "working"` | `"info"` | The semantic status of the toast. |

## Exports

- Components: `Toast`, `ToastActions`, `ToastContent`, `ToastIcon`, `ToastRoot`, `ToastText`
- Types: `ToastActionProps`, `ToastContentProps`, `ToastIconProps`, `ToastProps`, `ToastRootProps`, `ToastTextProps`

## See also

- [Toaster](./Toaster.md)
- [Notification](./Notification.md)
- [Banner](./Banner.md)
- [StatusMessage](./StatusMessage.md)
