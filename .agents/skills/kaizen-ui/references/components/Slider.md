# Slider

Also known as: Range Input, Input Range

A single-value range input rendered along a bar. Use for continuous settings like volume, brightness, or threshold values where the relative position carries meaning.

## Notes

- Use Slider when the relative position within a range matters more than the exact value. For precise numeric entry (port number, replica count, timeout) use TextInput with `type="number"` instead.

- Pair the Slider with a TextInput that displays and accepts the current value when the range is large (e.g. 0-10000), users may know their target value, or precision matters but the range context is still useful.

- Default to `orientation="horizontal"`. Reach for `orientation="vertical"` only when the metaphor is inherently vertical (audio fader, equalizer band, temperature).

- Show step marks (`stepPosition`) when the range has a small number of discrete valid values or when the step size is non-obvious. Hide them for continuous ranges or many fine-grained steps.

- Always render the slider's min and max values near the track so users understand the boundaries of the range.

- Set `step` to match the smallest meaningful increment so the thumb cannot land between valid values.

## Examples

### Basic Slider

```tsx
<Slider defaultValue={50} aria-label="Volume" />
```

### With Step Marks Slider

Use step marks to show discrete positions and help users pick precise values.

```tsx
<Slider defaultValue={40} step={10} stepPosition="end" aria-label="Brightness" />
```

### Custom Range Slider

Use custom min/max/step when the default 0–100 range does not match the domain.

```tsx
<Slider defaultValue={250} min={100} max={500} step={50} stepPosition="end" aria-label="Budget" />
```

### Disabled Slider

Use when the slider value is locked and cannot be changed.

```tsx
<Slider defaultValue={30} disabled aria-label="Locked" />
```

### Vertical Slider

Use vertical orientation for controls like volume faders or equalizer bands.

```tsx
<Slider defaultValue={60} orientation="vertical" stepPosition="start" step={20} aria-label="Level" />
```

### With Step Interval Slider

Use stepInterval to space the visible step markers at a coarser interval than the slider's underlying step value, so users can drag with fine precision while still seeing key reference points.

```tsx
<Slider defaultValue={50} min={0} max={100} step={1} stepInterval={20} stepPosition="end" aria-label="Precision control" />
```

### With Custom Steps Slider

Use customSteps when the meaningful tick positions are non-uniform or only certain anchor values matter (e.g. quality presets, named breakpoints). It overrides both step and stepInterval for marker placement.

```tsx
<Slider defaultValue={50} min={0} max={100} step={5} customSteps={[0, 25, 50, 75, 100]} stepPosition="end" aria-label="Quality preset" />
```

### With Formatted Steps Slider

Use stepFormatFn to render step labels in a domain-specific format like currency, percentages, units, or descriptive words (e.g. Mute / Quiet / Loud). Works with both orientations.

```tsx
<Slider defaultValue={60} min={0} max={100} step={1} stepInterval={20} stepPosition="end" stepFormatFn={(value) => `${value}°C`} aria-label="Temperature" />
```

### With Form Name Slider

Set name to submit the slider value as a standard form field — the underlying native range input participates in form submission with no extra wiring.

```tsx
<Slider name="volume" defaultValue={50} step={10} stepPosition="end" aria-label="Volume" />
```

### With Text Input Slider

Pair a Slider with a TextInput by sharing controlled state — useful when users need both quick coarse adjustment via the slider and precise numeric entry via the input. Track the numeric value alongside a string buffer so the user can type freely, then clamp the buffer into the slider's range on blur.

```tsx
() => {
  const min = 0;
  const max = 100;
  const [value, setValue] = useState(50);
  const [inputValue, setInputValue] = useState("50");

  return (
    <Flex align="center" gap="density-md">
      <TextInput
        className="w-fit"
        type="number"
        min={min}
        max={max}
        value={inputValue}
        onValueChange={(next) => {
          setInputValue(next);
          const parsed = Number(next);
          if (next.trim() !== "" && !Number.isNaN(parsed)) {
            const clamped = Math.min(max, Math.max(min, parsed));
            setValue(clamped);
          }
        }}
        onBlur={() => {
          const parsed = Number(inputValue);
          const clamped = Number.isNaN(parsed) ? value : Math.min(max, Math.max(min, parsed));
          setValue(clamped);
          setInputValue(String(clamped));
        }}
        aria-label="Volume"
      />
      <Slider
        min={min}
        max={max}
        value={value}
        onValueChange={(next) => {
          setValue(next);
          setInputValue(String(next));
        }}
        aria-label="Volume"
      />
    </Flex>
  );
};
```

### Composed

Use the composed primitives when you need full control over the slider track, input, and step label layout.

```tsx
<SliderRoot orientation="horizontal" aria-label="Custom slider">
  <SliderInput aria-label="Custom slider input" defaultValue={50} min={0} max={100} step={25} />
  <SliderSteps min={0} max={100} step={25} position="end" />
</SliderRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ SliderInput?: NativeElementAttributes<"input", typeof SliderInput>; SliderSteps?: NativeElementAttributes<"span", typeof SliderSteps> }` | - | Native HTML attributes forwarded to the internal composed components. |
| customSteps | `number[]` | - | Explicit list of values to render as markers. Overrides `step` and `stepInterval` when provided. |
| defaultValue | `number` | - | Initial value for uncontrolled use. |
| max | `number` | `100` | Upper bound of the value range. |
| min | `number` | `0` | Lower bound of the value range. |
| onValueChange | `(value: number) => void` | - | Fires on every value change while the user drags or steps the thumb. |
| onValueCommit | `(value: number) => void` | - | Fires once when the user releases the thumb. Useful for committing the final value to a backend. |
| orientation | `"horizontal" \| "vertical"` | `"horizontal"` | Whether the track runs horizontally or vertically. |
| step | `number` | `1` | Step increment between selectable values. |
| stepFormatFn | `(value: number) => string` | - | Formats each step label before it is rendered. Receives the numeric value and returns the display string. |
| stepInterval | `number` | - | Distance between rendered step markers. Falls back to the slider's `step` value when omitted. |
| stepPosition | `"start" \| "end"` | - | Side of the track the step markers render on. Markers are hidden when omitted. |
| value | `number` | - | Controlled value. Pair with `onValueChange` to keep the slider in sync. |

## Exports

- Components: `Slider`, `SliderInput`, `SliderRoot`, `SliderSteps`
- Types: `HorizontalSliderProps`, `SliderInputProps`, `SliderRootProps`, `SliderStepsProps`, `StepPosition`, `VerticalSliderProps`

## See also

- [RangeSlider](./RangeSlider.md)
