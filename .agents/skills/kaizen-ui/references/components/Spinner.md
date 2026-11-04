# Spinner

An indicator for short waits with unknown progress.

## Notes

- Name the work with `slotDescription`, or add an `aria-label` when a visible message will not fit.

- Use Skeleton when the content's shape is known.

- Use ProgressBar for longer work, even when progress is unknown.

- Use the default arrow for a section or page wait. Use `kind="ring"` beside text or in a table cell, where its smaller size and neutral color draw less attention.

- For `kind="default"` set `size` to match the context: `"small"` beside a control, `"medium"` (the default) inside a panel, and `"large"` when the spinner owns the whole view; it renders 32px, 64px, and 128px tall.

- For `kind="ring"` leave `size` unset so the ring scales with the surrounding text (`1em`); setting it fixes the ring at 12px, 16px, or 24px.

## Examples

### Basic Spinner

Use for a short wait when a visible message will not fit. Name the work with `aria-label` so screen readers have context.

```tsx
<Spinner aria-label="Fetching checkpoints" />
```

### Spinner With Description

Name the work with `slotDescription` when a visible message helps explain the wait. The description also labels the spinner for assistive technology.

```tsx
<Spinner slotDescription="Loading models..." />
```

### Ring Spinner

Use `kind="ring"` beside text or in a table cell, where its smaller size and neutral color draw less attention. Leave `size` unset so the ring scales with the surrounding text (`1em`).

```tsx
<div className="flex items-center gap-2">
  <Spinner kind="ring" aria-label="Checking availability" />
  <span>Checking availability...</span>
</div>
```

### Spinner Sizes

Use the small arrow beside text, medium for a section wait, and large when the wait occupies a whole view. Leave `size` unset for the ring to scale with surrounding text (`1em`); setting it fixes the ring at 12px, 16px, or 24px.

```tsx
<div className="flex flex-col items-start gap-4">
  <Spinner size="small" aria-label="Loading" />
  <Spinner size="medium" aria-label="Loading" />
  <Spinner size="large" aria-label="Loading" />
</div>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **slotDescription** * | `ReactNode` | - | Message rendered below the spinner. Also labels the spinner for assistive technology, so an `aria-label` is not required when set. |
| aria-label | `string` | - | Defines a string value that labels the current element. Accessible label. Omit when `slotDescription` is provided. |
| ~~description~~ | `ReactNode` | - | Message rendered below the spinner that also labels it for assistive technology. *(Deprecated: Use `slotDescription` instead.)* |
| kind | `"default" \| "ring"` | `"default"` | Which indicator to draw. `"default"` is the NVIDIA arrow mark; `"ring"` is a small rotating ring. |
| size | `"small" \| "medium" \| "large"` | `"medium" for the `"default"` kind, `1em` for `"ring"`` | Visual size of the spinner. Each kind has its own scale: the `"default"` kind renders 32px, 64px, and 128px tall (and proportionally wider), the square `"ring"` kind 12px, 16px, and 24px. Leave `size` unset to scale the ring with the surrounding text (`1em`). |

`* = required prop`

## Exports

- Components: `Spinner`
- Types: `SpinnerProps`

## See also

- [Skeleton](./Skeleton.md)
- [ProgressBar](./ProgressBar.md)
