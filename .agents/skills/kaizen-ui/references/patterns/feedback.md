# Feedback

How the system communicates the result of user actions or surfaces important information. Feedback is proportional to the significance of the event — critical errors demand attention, routine confirmations should not interrupt.

## When to use

Use a feedback component when:

- The action's result is not on the current screen (e.g., "Deployment created" but the user is still on the creation form)
- The action's result might be missed in a busy interface (e.g., a new item added to a long table — toast + scroll to or highlight the new row)
- A page-wide announcement or persistent context needs to be surfaced
- An asynchronous operation continues after the user moves on, or completes once they have

Don't use when:

- The UI change is itself sufficient feedback (a row appears, a modal closes, content refreshes)
- The action is so routine that feedback adds noise (e.g., navigating between pages, closing a panel, scrolling)
- The user explicitly requested the outcome and the outcome is visible (e.g., after "Clear Filters," the unfiltered list is the feedback)

Errors have a dedicated decision — see [error-states.md](./error-states.md). For destructive / high-stakes actions, use a modal dialog or inline confirmation.

---

## Variants

Pick the surface by severity and persistence.

| Variant | Use when | Reference |
| --- | --- | --- |
| **Toast** | Success / minor info, auto-dismissable | [Toast.md](../components/Toast.md) |
| **Banner** | Page-wide announcement / persistent context | [Banner.md](../components/Banner.md) |
| **Notification** | Async detailed update with optional action | [Notification.md](../components/Notification.md) |
| **Error surfaces** | Errors (page / container / inline / transient) | [error-states.md](./error-states.md) owns the full decision |
| **Confirmations** | Destructive / high-stakes actions | Use a modal dialog or inline confirmation |


Use [Toaster](../components/Toaster.md) for feedback over the page. It owns placement, stacking, timers, and dismissal.



**Anti-patterns:**

- Don't toast errors that need a decision or must not be missed — those need durable, in-context surfaces. A **recoverable, transient** failure may use a `Toast status="error"` carrying a retry action; [error-states.md](./error-states.md) owns the full Toast-vs-Notification-vs-Banner decision.
- Don't use a modal dialog where a Banner or inline error would work


Render a `Toaster` near the app root and call `toast()` to add feedback.







---

## Composition rules

Apply to all variants.

### Severity tone

Use the component's `status` prop — do not hardcode colors.

### Double-signal when result is off-screen

In the two off-screen cases under [When to use](#when-to-use) (result not on the current screen; result easily missed in a busy interface), pair the feedback component **with** a visible UI change (e.g., toast + scroll-to/highlight the new row) rather than relying on either alone.

---

## Modifiers

### Asynchronous / background

For operations that continue after the user moves on:

- Show a Toast confirming the action was initiated (e.g., "Export started. We'll notify you when it's ready.")
- Show a floating Notification when the operation completes
- If the application has a task queue or activity view, update it with the operation status
