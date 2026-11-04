# Loading

Loading indicators communicate system status and set user expectations about wait time. The wrong loading pattern can make a fast system feel slow or a slow system feel broken.

## When to use

Show an indicator once an operation exceeds ~300ms; below that, show nothing (see [Timing thresholds](#timing-thresholds) for the full scale). Skip the indicator entirely when optimistic UI fits.

---

## Anatomy

A visual indicator (spinner / skeleton / progress bar) plus optional context that aids the wait. The non-obvious requirement:

- **Reserve the indicator's space** to match the final content's dimensions, to minimize layout shift when content resolves.

### Optional

Include when applicable.

| Part | Include when |
| --- | --- |
| **Label / context message** | Operation is long enough that the user needs to know what is happening ("Processing 1,247 records…", "Training model — this may take a minute") |
| **Percentage / step count** | Progress is determinate and the value is meaningful to the user |
| **Estimated time remaining** | Wait is expected to exceed 60 seconds |
| **Cancel / dismiss action** | Operation is safely cancellable and the user may want to abandon it |
| **Retry action** | Loading failed and the failure is likely transient (see [Composition rules → Loading errors](#loading-errors)) |

---

## Variants

Pick the indicator by whether the wait duration is known and whether the content layout is predictable. For how the treatment escalates with wait length, see [Timing thresholds](#timing-thresholds).

| Variant | Use when |
| --- | --- |
| **Progress bar (determinate)** | System knows how far along the process is (% or steps complete); operation has measurable start + end |
| **Spinner (indeterminate)** | System cannot predict duration; content layout is NOT predictable |
| **Skeleton screen** | Content layout IS predictable (cards, table rows, text blocks) |
| **Button loading (inline spinning icon)** | User clicked a button to trigger an async action |
| **Optimistic UI** | High-success-rate, low-stakes action with a predictable result (no indicator shown) |

### Progress bar (determinate)

For uploads, exports, training, multi-step pipelines. See the [`ProgressBar`](../components/ProgressBar.md).

### Indeterminate spinner

Center it in the container if the whole view is loading; place it inline within the trigger with the `small` size variant if a single component is loading.

### Skeleton screen

Best for predictable/standardized layouts — the allocated space prevents layout shift. See the [`Skeleton`](../components/Skeleton.md).

### Button loading (inline spinner)

Use a small, non-distracting spinner inside the triggering button. Retain a meaningful accessible name, expose a busy state, and prevent duplicate submission while the operation is active.


```tsx
<Button loading={inProgress}>
  {inProgress ? "Downloading..." : "Download"}
</Button>
```







### Optimistic UI

Show the result immediately and reconcile when the server responds — for high-success-rate, low-stakes, predictable actions (toggle a setting, mark as read, add a tag). On rejection, revert the UI atomically and surface an error. Never for destructive, financial, or unpredictable actions.

---

## Composition rules

Apply to all variants.

### Progressive loading: chrome → structure → content

Isolate loading states to narrow content areas to allow for progressive loading, so pages aren't gated on the slowest endpoint.

### Avoid full-page spinners

Don't cover the whole page with one spinner if regions can render independently — it feels slower and conveys nothing. **Exception:** auth screens, initial app boot, or true full-page transitions where no content exists yet.

### Loading errors

Replace the indicator with an error state (see [`error-states.md`](./error-states.md)). Scope the error to the failed region rather than blocking the whole page, and offer retry only when the failure is likely transient.

---

## Modifiers

### Timing thresholds

Match the indicator (or absence of one) to the expected wait duration.

| Duration | Treatment |
| --- | --- |
| **Under 300ms** | Show nothing. The response feels instant; any indicator creates a flash that feels more jarring than the wait. |
| **300ms – 1s** | Show a subtle indicator only if you can do so without a flash. Skeleton screens work well — they're already part of the layout. Delay-mount spinners by 300ms so they never appear if content arrives first. |
| **1 – 10s** | Show the appropriate indicator (see [Variants](#variants)). This is the core range where indicators are essential. |
| **10 – 60s** | Show a progress indicator (determinate if possible) with context about what's happening ("Processing 1,247 records…", "Training model — this may take a minute"). |
| **Over 60s** | Show a progress indicator with estimated time remaining. Consider allowing the user to navigate away and return — surface task status in a persistent location (a notification, a status bar, or a task queue view). Notify the user when the long-running task completes. |
