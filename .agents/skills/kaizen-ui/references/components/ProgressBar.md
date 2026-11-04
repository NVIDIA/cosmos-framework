# ProgressBar

Visual indicator for long-running operations like uploads, downloads, and other loading states.

## Notes

- Pass `aria-label` or `aria-labelledby` so assistive tech can announce what the bar represents.

- Use Spinner (not ProgressBar) for indeterminate loading where no percentage exists — `kind="indeterminate"` is rarely the right choice when a Spinner would do.

- Use Skeleton (not ProgressBar) when the content being loaded has a known layout — skeletons reduce perceived load time more than a bar does.

## Examples

### Basic Progress Bar

Use a determinate progress bar when the completion percentage is known.

```tsx
<ProgressBar aria-label="65% complete" value={65} />
```

### Indeterminate Progress Bar

Use an indeterminate progress bar when the duration of a process is unknown.

```tsx
<ProgressBar kind="indeterminate" aria-label="Loading..." />
```

### Sizes Progress Bar

Match the bar to its surroundings: `small` for inline or table row indicators, `medium` (default) for section-level progress, and `large` for prominent file uploads or page-level loading.

```tsx
<div className="flex flex-col gap-4">
  <ProgressBar aria-label="30% complete" value={30} size="small" />
  <ProgressBar aria-label="55% complete" value={55} size="medium" />
  <ProgressBar aria-label="80% complete" value={80} size="large" />
</div>
```

### Labeled Progress Bar

Pair the progress bar with a visible text label using `aria-labelledby` when users benefit from seeing the operation name and percentage alongside the bar.

```tsx
<div className="w-full">
  <div className="mb-1 flex justify-between">
    <span id="progress-bar-label">Uploading documents</span>
    <span aria-hidden="true">50%</span>
  </div>
  <ProgressBar aria-labelledby="progress-bar-label" value={50} />
</div>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **aria-label** * | `string` | - | Accessible name that tells assistive technology what the bar tracks. Provide this or `aria-labelledby`. |
| **value** * | `number` | - | Current progress as a percentage from 0 to 100. Values above 100 render a full bar. |
| aria-labelledby | `string` | - | Id of a visible element that names the bar for assistive technology. Provide this or `aria-label`. |
| kind | `"determinate" \| "indeterminate"` | `"determinate"` | Whether the bar reflects a known percentage or an unknown duration. - "determinate" - Fills to `value` and is suited for known-length operations like file uploads. - "indeterminate" - Animates continuously and is suited for operations with unknown duration. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Size of the progress bar. |

`* = required prop`

## Exports

- Components: `ProgressBar`
- Types: `ProgressBarProps`

## See also

- [Spinner](./Spinner.md)
- [Skeleton](./Skeleton.md)
