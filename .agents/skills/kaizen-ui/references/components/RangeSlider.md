# RangeSlider

Also known as: Range Input, Input Range

A two-thumb slider for selecting a minimum and maximum value within a range. Use for filtering by price, time window, or any min/max selection.

## Notes

- Always provide `aria-label` (or `aria-labelledby`); each thumb derives its accessible name from it.

## Examples

### Basic Range Slider

```tsx
<RangeSlider defaultValue={[25, 75]} aria-label="Price range" />
```

### With Step Marks Range Slider

Use step marks to show discrete positions and help users pick precise range boundaries.

```tsx
<RangeSlider defaultValue={[20, 80]} step={10} stepPosition="end" aria-label="Budget range" />
```

### With Formatted Steps Range Slider

Use stepFormatFn to render step labels in a domain-specific format like currency, percentages, or units.

```tsx
<RangeSlider defaultValue={[200, 800]} min={0} max={1000} step={100} stepPosition="end" stepFormatFn={(value) => `$${value}`} aria-label="Price range" />
```

### With Custom Steps Range Slider

Use customSteps when the meaningful values are non-uniform, like pricing tiers or named breakpoints.

```tsx
<RangeSlider defaultValue={[100, 400]} min={0} max={500} step={50} stepPosition="end" customSteps={[0, 100, 250, 500]} aria-label="Pricing tier range" />
```

### Custom Range Range Slider

Use custom min/max/step when the default 0–100 range does not match the domain.

```tsx
<RangeSlider defaultValue={[200, 800]} min={0} max={1000} step={100} stepPosition="end" aria-label="Price filter" />
```

### With Min Distance Range Slider

Use minStepsBetweenThumbs to enforce a minimum gap between the two thumbs, preventing collapsed selections.

```tsx
<RangeSlider defaultValue={[20, 80]} step={5} stepPosition="end" minStepsBetweenThumbs={4} aria-label="Date range" />
```

### Disabled Range Slider

Use when the range selection is locked and cannot be changed.

```tsx
<RangeSlider defaultValue={[30, 70]} disabled aria-label="Locked range" />
```

### Vertical Range Slider

Use vertical orientation for controls like equalizer bands or vertical filtering.

```tsx
<RangeSlider defaultValue={[20, 60]} orientation="vertical" stepPosition="start" step={20} aria-label="Level range" />
```

### With Form Name Range Slider

Set name to submit the range as two repeated form fields, supporting both hydrated and no-JS form submissions via the native fallback inputs.

```tsx
<RangeSlider name="priceRange" defaultValue={[100, 300]} min={0} max={500} step={25} stepPosition="end" aria-label="Price range" />
```

### Composed

Use the composed primitives when you need full control over track, range, thumb, and step label layout.

```tsx
<RangeSliderRoot defaultValue={[25, 75]} min={0} max={100} step={25} aria-label="Custom range">
  <RangeSliderTrack>
    <RangeSliderRange />
  </RangeSliderTrack>
  <RangeSliderThumb aria-label="Minimum" />
  <RangeSliderThumb aria-label="Maximum" />
  <RangeSliderSteps position="end" />
</RangeSliderRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ RangeSliderTrack?: NativeElementAttributes<"span", typeof RangeSliderTrack>; RangeSliderRange?: NativeElementAttributes<"span", typeof RangeSliderRange>; RangeSliderThumbStart?: NativeElementAttributes<"span", typeof RangeSliderThumb>; RangeSliderThumbEnd?: NativeElementAttributes<"span", typeof RangeSliderThumb>; RangeSliderSteps?: NativeElementAttributes<"span", typeof RangeSliderSteps>; RangeSliderNativeFallbackFrom?: NativeElementAttributes<"input", typeof RangeSliderNativeFallback>; RangeSliderNativeFallbackTo?: NativeElementAttributes<"input", typeof RangeSliderNativeFallback> }` | - | Native HTML attributes forwarded to the internal composed components. |
| customSteps | `number[]` | - | Explicit list of step values to render, overriding `stepInterval`. |
| defaultValue | `[number, number]` | - | Initial `[start, end]` value when uncontrolled. |
| disabled | `boolean` | `false` | Disables the slider. |
| form | `string` | - | `id` of the form the slider belongs to. Defaults to the closest ancestor `<form>`. |
| max | `number` | `100` | Maximum selectable value. |
| min | `number` | `0` | Minimum selectable value. |
| minStepsBetweenThumbs | `number` | - | Minimum number of steps that must separate the two thumbs. |
| name | `string` | - | HTML `name` submitted with the form. Falls back to the surrounding `FormField`'s `name` when omitted. |
| onValueChange | `(value: [number, number]) => void` | - | Called continuously as the user drags a thumb. |
| onValueCommit | `(value: [number, number]) => void` | - | Called once the user finishes interacting (pointer release or keyboard commit). Useful for syncing with a backend without firing on every intermediate value. |
| orientation | `"horizontal" \| "vertical"` | `"horizontal"` | The orientation of the slider. |
| step | `number` | `1` | Step granularity between adjacent selectable values. |
| stepFormatFn | `(value: number) => string` | - | Formats each step label before it is rendered. |
| stepInterval | `number` | - | Spacing between rendered step markers. Defaults to the slider's `step`. |
| stepPosition | `"start" \| "end"` | `undefined` | The position of the step markers. |
| value | `[number, number]` | - | Controlled `[start, end]` value. When provided, the consumer is responsible for updating it in response to `onValueChange`. |

## Exports

- Components: `RangeSlider`, `RangeSliderNativeFallback`, `RangeSliderRange`, `RangeSliderRoot`, `RangeSliderSteps`, `RangeSliderThumb`, `RangeSliderTrack`
- Utilities: `clamp`, `convertValueToPercentage`, `formatStep`, `generateSteps`, `normalizeRangeSliderValue`, `parseRangeSliderFromFormData`, `validateRangeSliderValue`
- Types: `RangeSliderHorizontalProps`, `RangeSliderNativeFallbackProps`, `RangeSliderProps`, `RangeSliderRangeProps`, `RangeSliderRootProps`, `RangeSliderStepPosition`, `RangeSliderStepsProps`, `RangeSliderThumbProps`, `RangeSliderTrackProps`, `RangeSliderValidationError`, `RangeSliderValue`, `RangeSliderVerticalProps`
