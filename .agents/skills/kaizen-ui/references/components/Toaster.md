# Toaster

Renders and manages the app's toast and notification stacks. Render one near
the app root, then call `toast()` or `notification()` from anywhere.

## Notes

- Keep the `position.toast` and `position.notification` entries different
  so the two surfaces never compete for the same viewport space.

- Render exactly one `Toaster`. A second one duplicates every item because
  both read the same queues.

- Call `toast()` and `notification()` from an event handler or effect.
  Feedback raised while rendering on the server is dropped because it cannot be shown there.

- Items render in the top layer and move inside an open Kaizen UI modal so
  their actions remain clickable.

## Examples

### Basic Toaster

Render one `Toaster` near the app root and call `toast()` from anywhere, including code outside React.

```tsx
<>
  <Button onClick={() => toast("Settings saved", { status: "success" })}>Save settings</Button>
  <Toaster />
</>
```

### Notification Toaster

Use `notification()` when peripheral feedback needs supporting detail or an action; use `toast()` for brief confirmations.

```tsx
<>
  <Button
    onClick={() =>
      notification("Update available", {
        footer: <Button size="small">Install update</Button>,
        status: "info",
        subheading: "Restart the application after installation.",
      })
    }
  >
    Check for updates
  </Button>
  <Toaster />
</>
```

### Toaster With Undo

Pass `action` to offer a single follow-up, such as undoing a destructive action or jumping to a result. Run the follow-up from the action's `onClick` and dismiss the toast with the id `toast()` returned, so it stops offering an undo that has already happened. Leave routine confirmations without an action.

```tsx
<>
  <Button
    color="danger"
    onClick={() => {
      const id = toast("Deleted 3 items", {
        status: "neutral",
        action: (
          <Button kind="tertiary" onClick={() => toast.dismiss(id)} size="small">
            Undo
          </Button>
        ),
      });
    }}
  >
    Delete items
  </Button>
  <Toaster />
</>
```

### Promise Toaster

`toast.promise` shows a `working` toast while the promise is pending, then replaces it in place with the result. The same toast changes status rather than a second one appearing.

```tsx
<>
  <Button
    onClick={() =>
      toast.promise(new Promise<string>((resolve) => setTimeout(() => resolve("draft"), 1500)), {
        loading: "Saving",
        success: (name) => `Saved ${name}`,
        error: "Could not save",
      })
    }
  >
    Save draft
  </Button>
  <Toaster />
</>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| accessibleNames | `{ notification?: string; toast?: string }` | `{ notification: "Detailed notifications", toast: "Notifications" }` | Accessible names of the toast and notification regions. |
| closeButton | `boolean` | `true` | Renders a close button on items that do not opt out with `dismissible: false`. |
| duration | `number` | `5000` | Default time an item stays on screen, in milliseconds. Individual items override it with their own `duration`. |
| forceExpand | `boolean` | `false` | Keeps both stacks expanded instead of allowing interaction to control their collapsed state. |
| gap | `number` | `8` | Space between items once a stack is expanded, in pixels. |
| hotkey | `readonly string[]` | `["altKey", "KeyT"]` | Keys that move focus into the toast stack. Each entry is a `KeyboardEvent` modifier property or a `KeyboardEvent.code`. |
| max | `number` | `3` | How many items from each stack are on screen at once. Older items wait off screen, keep counting down, and reappear if space opens before they expire. |
| offset | `number` | `24` | Distance from the viewport edge, in pixels. |
| portal | `boolean` | `false` | Renders the floating content in a React portal. |
| portalContainer | `HTMLElement` | `document.body` | Container for portaled floating content when no nearer portal target exists. |
| position | `{ notification: "top-left" \| "top-center" \| "top-right" \| "bottom-left" \| "bottom-center" \| "bottom-right"; toast: "top-left" \| "top-center" \| "top-right" \| "bottom-left" \| "bottom-center" \| "bottom-right" }` | `DEFAULT_TOASTER_POSITION` | Positions of the toast and notification stacks. The two entries must differ. |
| width | `string \| number` | - | Width of each toast and notification. Numbers are treated as pixels. |

## Exports

- Components: `Toaster`
- Hooks: `useToaster`
- Utilities: `notification`, `toast`
- Constants: `TOASTER_POSITIONS`
- Types: `NotificationKind`, `NotificationOptions`, `NotificationRecord`, `NotificationStatus`, `ToastDismissReason`, `ToasterAccessibleNames`, `ToasterPosition`, `ToasterPositions`, `ToasterProps`, `ToasterRecord`, `ToastId`, `ToastOptions`, `ToastPromiseOptions`, `ToastRecord`, `ToastRender`, `ToastRenderProps`, `ToastStatus`

## See also

- [Toast](./Toast.md)
- [Notification](./Notification.md)
