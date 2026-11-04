# Notification

A non-blocking overlay message that delivers detailed feedback in the user's peripheral vision, optionally with actions. Reach for it when the message needs more detail than a Toast or an action the user should take.

## Notes

- Use Toast (not Notification) for brief confirmations ("Saved", "Copied") — Notification is for detailed messages with optional actions.

- Use Banner (not Notification) for inline page-level or system-wide announcements — Notification is an overlay in the user's peripheral vision.

- Always set `status` so users can assess severity at a glance.

- Persistent notifications (no auto-dismiss) should be reserved for messages that require user action; informational notifications should auto-dismiss or be user-dismissible via `onClose`.

- Pick the messaging surface by attention and persistence: Toast (lowest attention, transient) → Notification (medium attention, peripheral, stackable) → Banner (high attention, inline, persistent) → Modal (highest attention, blocking). Overusing high-attention or persistent surfaces leads to alert fatigue.

- Notification vs Toast — Toast is brief (≤4 words), queues one at a time, and auto-dismisses; Notification stacks, supports heading + subheading + footer, and can persist.

- When shown through Toaster, Notification defaults to top-center. Keep its placement distinct from Toast's placement.

- Watch stacking: when notifications stack, their combined attention cost compounds. Consider whether stacked notifications are truly independent or should be consolidated into a single summary.

- Frame temporary vs persistent by behavior — "Job completed" auto-dismisses; "Update available — upgrade now" persists until the user acts. Persistent notifications can still offer `onClose` when the action can be deferred.

- `kind="stacked"` is the default and gives more room for subheading and footer content; switch to `kind="inline"` only for compact single-line notifications.

## Examples

### Basic Notification

Use for an informational message that should auto-dismiss after a short delay or be paired with `onClose` so the user can clear it themselves.

```tsx
<Notification status="info" slotSubheading="Check your inbox for details.">
  New message received
</Notification>
```

### Dismissible Notification

Pass `onClose` to render the dismiss button — use whenever the user should be able to clear the notification themselves.

```tsx
<Notification status="success" slotSubheading="Your model is ready for inference." onClose={handleClose}>
  Deployment complete
</Notification>
```

### Statuses Notification

Set `status` to convey severity: `info` for context, `success` for confirmation, `warning` for caution, and `error` for critical failures that need a next step.

```tsx
<Flex direction="col" gap="density-md">
  <Notification status="info" slotSubheading="Added 50K labeled images to the dataset.">
    New training data available
  </Notification>
  <Notification status="success" slotSubheading="Inference endpoint active at api.nvidia.com/v1/chat.">
    Model deployed successfully
  </Notification>
  <Notification status="warning" slotSubheading="GPU 0: 92% memory utilization — consider reducing batch size.">
    High GPU memory usage detected
  </Notification>
  <Notification status="error" slotSubheading="Service temporarily unavailable — engineers investigating.">
    Inference endpoint unavailable
  </Notification>
</Flex>
```

### With Footer Notification

Pass `slotFooter` with action buttons when the notification needs an inline follow-up action like view, retry, or dismiss.

```tsx
<Notification
  status="warning"
  slotSubheading="3 resources are over budget this month."
  slotFooter={
    <>
      <Button kind="secondary" size="small">
        View Details
      </Button>
      <Button kind="tertiary" size="small">
        Dismiss
      </Button>
    </>
  }
>
  Budget alert
</Notification>
```

### With Icon Notification

Override `slotIcon` when a domain-specific glyph communicates the message better than the default status icon.

```tsx
<Notification status="info" slotIcon={<Bell />} slotSubheading="Version 2.1.0 is available." onClose={handleClose}>
  Update available
</Notification>
```

### Inline Notification

Use `kind="inline"` for compact single-line notifications, especially when several stack together and the default `stacked` layout would feel oversized. For form/page-level validation, use Banner or FormField's `slotError` instead.

```tsx
<Flex direction="col" gap="density-sm">
  <Notification kind="inline" status="success">
    Sync complete
  </Notification>
  <Notification kind="inline" status="info">
    3 background jobs running
  </Notification>
  <Notification kind="inline" status="warning">
    Token expires in 5 minutes
  </Notification>
</Flex>
```

### Composed

Use composed primitives when you need full control over the notification grid layout and element placement.

```tsx
<NotificationRoot status="success">
  <NotificationContent>
    <NotificationIcon status="success" />
    <NotificationHeader>
      <NotificationHeading>Composed notification</NotificationHeading>
      <NotificationSubheading>Using primitives for full layout control.</NotificationSubheading>
    </NotificationHeader>
    <NotificationFooter>
      <Button kind="secondary" size="small">
        View
      </Button>
    </NotificationFooter>
  </NotificationContent>
</NotificationRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| attributes | `{ NotificationContent?: NativeElementAttributes<"div", typeof NotificationContent>; NotificationIcon?: NativeElementAttributes<"div", typeof NotificationIcon>; NotificationHeader?: NativeElementAttributes<"div", typeof NotificationHeader>; NotificationHeading?: NativeElementAttributes<"span", typeof NotificationHeading>; NotificationSubheading?: NativeElementAttributes<"div", typeof NotificationSubheading>; NotificationCloseButton?: NativeElementAttributes<"button", typeof Button>; NotificationFooter?: NativeElementAttributes<"div", typeof NotificationFooter> }` | - | Native HTML attributes forwarded to the internal composed components. |
| children | `ReactNode` | - | Primary message of the notification. |
| kind | `"inline" \| "stacked"` | `"stacked"` | Layout density of the notification. - "stacked" - Heading, subheading, and footer stack vertically. - "inline" - Heading and subheading sit on a single line. |
| onClose | `() => void` | - | Dismiss handler. When provided, a close button is rendered. |
| ref | `Ref<HTMLDivElement>` | - |  |
| slotCloseIcon | `ReactNode` | `<Icon name="close" />` | Custom icon for the close button. Only renders when `onClose` is provided. |
| slotFooter | `ReactNode` | - | Footer content rendered below the message, typically used for action buttons. |
| slotIcon | `ReactNode` | - | Icon rendered alongside the heading. Defaults to a status-appropriate icon; pass `null` to render no icon. |
| slotSubheading | `ReactNode` | - | Supporting text rendered below the heading. |
| status | `"error" \| "warning" \| "success" \| "info"` | `"info"` | The semantic status of the notification. - "info" - Additional information that may require action. - "success" - Confirms a completed action or task. - "warning" - Cautions the user about something urgent to avoid an issue. - "error" - Communicates a critical issue that needs attention. |

## Exports

- Components: `Notification`, `NotificationContent`, `NotificationFooter`, `NotificationHeader`, `NotificationHeading`, `NotificationIcon`, `NotificationRoot`, `NotificationSubheading`
- Types: `NotificationContentProps`, `NotificationFooterProps`, `NotificationHeaderProps`, `NotificationHeadingProps`, `NotificationIconProps`, `NotificationProps`, `NotificationRootProps`, `NotificationSubheadingProps`

## See also

- [Toast](./Toast.md)
- [Banner](./Banner.md)
- [StatusMessage](./StatusMessage.md)
