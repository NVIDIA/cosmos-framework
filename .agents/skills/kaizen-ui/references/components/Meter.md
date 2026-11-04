# Meter

A bar showing how much of a bounded resource is in use.

## Notes

- Use Meter (not ProgressBar) for a static reading like disk usage or quota consumption. ProgressBar announces itself as an operation in progress.

- Use SegmentedMeter (not Meter) to break a quantity down by category. Meter reads one value; it has no notion of segments.

- `max` defaults to 1, exactly as on the element — pass it whenever the reading is not already a fraction, or the bar renders full.

- `low`, `high` and `optimum` color the bar. Pass all three: `optimum` decides which end of the scale is the good one, and left unset it sits at the midpoint, where both outer bands read as merely off and the bar never reaches the danger color at all.

- For a reading that is worse as it climbs — disk usage, quota, error rate — pass `optimum={0}`. For one that is better as it climbs, such as capacity remaining, pass `optimum={max}`.

- `low` and `high` are in the same units as `value` and `max`, not percentages. On `max={500}` the thresholds are numbers like `300` and `425`; passing `60` and `85` puts both near the floor and paints every meter red.

- Wrap the meter in a `FormField` to label and describe it. Meter has no label of its own, so a bare one needs an `aria-label`.

- Change the color by setting the `--nv-meter-color` custom property.

## Examples

### Single Meter

Meter is a native `<meter>` element wearing the design system's styling. `value`, `min` and `max` mean here exactly what they mean there — including `max` defaulting to 1, so a reading that is not already a fraction needs one. Meter carries no label of its own, so wrap it in a `FormField`: because a meter is a labelable element, the field's label names it the way it would an input.

```tsx
<FormField slotLabel="Tasks complete" slotLabelEnd={<span className="ml-auto">2 of 4</span>}>
  <Meter max={4} value={2} />
</FormField>
```

### Quota Thresholds Meter

The common dashboard case: a reading that is _worse_ as it climbs. `optimum={0}` is what says so — it puts the good end of the scale at the bottom, so the bar goes warning and then danger as it fills. Note that `low` and `high` are in the same units as `value` and `max`, not percentages: on a 500 GB quota the thresholds are `300` and `425`, not `60` and `85`. Passing percentages to a scale that is not 0–100 puts both thresholds near the floor and paints every meter red.

```tsx
<div className="flex w-full flex-col gap-4">
  <Meter aria-label="118 GB of 500 GB used" high={425} low={300} max={500} optimum={0} value={118} />
  <Meter aria-label="360 GB of 500 GB used" high={425} low={300} max={500} optimum={0} value={360} />
  <Meter aria-label="482 GB of 500 GB used" high={425} low={300} max={500} optimum={0} value={482} />
</div>
```

### Thresholds Meter

`low` and `high` cut the scale into three bands and the browser puts the reading in one of them: the bar keeps its normal fill in the optimum band, turns the warning color in the one beside it, and the danger color beyond that. `optimum` decides which end is the good one — these read storage remaining, so it sits at the top. Left unset it defaults to the middle, and while it sits between `low` and `high` both outer bands count as merely off; nothing reaches the danger color until you push it to one end.

```tsx
<div className="flex w-full flex-col gap-4">
  <Meter aria-label="Plenty left" high={80} low={20} max={100} optimum={100} value={92} />
  <Meter aria-label="Getting low" high={80} low={20} max={100} optimum={100} value={50} />
  <Meter aria-label="Almost gone" high={80} low={20} max={100} optimum={100} value={8} />
</div>
```

### Min Meter

`min` moves the bottom of the scale, so the reading is drawn as its share of `min`–`max` rather than of zero–`max`.

```tsx
<Meter aria-label="Temperature" max={40} min={-10} value={18} />
```

### Sizes Meter

Match the bar to its surroundings: `small` for inline or table row readings, `medium` (default) for section-level meters, and `large` when the meter is the focus of a card or tile.

```tsx
<div className="flex w-full flex-col gap-4">
  <Meter aria-label="Small" max={100} size="small" value={65} />
  <Meter aria-label="Medium" max={100} size="medium" value={65} />
  <Meter aria-label="Large" max={100} size="large" value={65} />
</div>
```

### Custom Color Meter

Every fill is a custom property, so there is no `color` prop to learn: set `--nv-meter-color` for the normal bar, or `--nv-meter-suboptimum-color` and `--nv-meter-sub-sub-optimum-color` for the threshold bands. Set them wherever suits — inline as here, or in a rule of your own so a whole region of the app picks them up at once.

```tsx
<Meter aria-label="Cluster capacity" max={100} style={{ "--nv-meter-color": "var(--color-chart-data-categorical-3)" } as never} value={72} />
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Height of the bar. |

## Exports

- Components: `Meter`
- Types: `MeterProps`, `MeterSize`

## See also

- [SegmentedMeter](./SegmentedMeter.md)
- [FormField](./FormField.md)
- [ProgressBar](./ProgressBar.md)
