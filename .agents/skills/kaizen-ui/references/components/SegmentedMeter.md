# SegmentedMeter

A bar showing how a bounded quantity divides across categories.

## Notes

- Prefer Meter (not SegmentedMeter) for a single reading.

- Render a `Legend` next to a SegmentedMeter whenever its colors distinguish categories, and mark the meter `decorative`.

- Write the Legend's `items` in the same order as the meter's.

## Examples

### Basic Segmented Meter

Pair a segmented meter with a `Legend`. The bars carry no text, so the legend is what names each category and says what it is worth — the difference between a bar someone can read and a row of colors they can only guess at. Both need it: a screen reader user, because the meter is hidden from assistive technology, and a sighted user, because 28.3 GB is not something you can eyeball off a stripe. Write the Legend's `items` in the same order as the meter's, since both take the categorical data colors by position.

```tsx
<FormField labelable={false} slotLabel="Internal storage" slotLabelEnd={<span className="ml-auto">106.7 GB / 256 GB</span>}>
  <SegmentedMeter decorative max={256} items={[{ value: 43.3 }, { value: 28.3 }, { value: 20.2 }, { value: 14.9 }]} size="large" />
  <Legend
    items={[
      { children: "Videos", slotEnd: "43.3 GB" },
      { children: "Images", slotEnd: "28.3 GB" },
      { children: "Documents", slotEnd: "20.2 GB" },
      { children: "Audio files", slotEnd: "14.9 GB" },
    ]}
  />
</FormField>
```

### Share Of Total Segmented Meter

Leave `max` unset to show the items as shares of their own total. The track always fills completely, so no unused capacity is implied. A horizontal legend suits a short breakdown like this one.

```tsx
<FormField labelable={false} slotLabel="Browser share">
  <SegmentedMeter decorative items={[{ value: 64 }, { value: 19 }, { value: 17 }]} />
  <Legend
    items={[
      { children: "Chrome", slotEnd: "64%" },
      { children: "Safari", slotEnd: "19%" },
      { children: "Firefox", slotEnd: "17%" },
    ]}
    orientation="horizontal"
  />
</FormField>
```

### Sizes Segmented Meter

Match the track to its surroundings: `small` for inline or table row readings, `medium` (default) for section-level meters, and `large` when the meter is the focus of a card or tile. These are labelled rather than paired with a legend, since the two bars are tiers of one quantity rather than categories to tell apart.

```tsx
<Stack className="w-full" gap="density-xl">
  <SegmentedMeter aria-label="Small — 65 of 100 GB used" max={100} items={[{ value: 45 }, { value: 20 }]} size="small" />
  <SegmentedMeter aria-label="Medium — 65 of 100 GB used" max={100} items={[{ value: 45 }, { value: 20 }]} size="medium" />
  <SegmentedMeter aria-label="Large — 65 of 100 GB used" max={100} items={[{ value: 45 }, { value: 20 }]} size="large" />
</Stack>
```

### Custom Colors Segmented Meter

Items default to the categorical data colors in order. Set `color` to pin a category that has an established meaning, and the same value on the matching legend entry — it is the same prop on both sides, so one array of series can feed the pair without being reshaped.

```tsx
<FormField labelable={false} slotLabel="Cluster health">
  <SegmentedMeter
    decorative
    max={100}
    items={[
      { color: "var(--color-chart-data-categorical-7)", value: 12 },
      { color: "var(--color-chart-data-categorical-3)", value: 24 },
      { color: "var(--color-chart-data-categorical-1)", value: 58 },
    ]}
  />
  <Legend
    items={[
      {
        children: "Critical",
        color: "var(--color-chart-data-categorical-7)",
        slotEnd: "12 nodes",
      },
      {
        children: "Degraded",
        color: "var(--color-chart-data-categorical-3)",
        slotEnd: "24 nodes",
      },
      {
        children: "Healthy",
        color: "var(--color-chart-data-categorical-1)",
        slotEnd: "58 nodes",
      },
    ]}
  />
</FormField>
```

### Composed Segmented Meter

```tsx
<FormField labelable={false} slotLabel="Internal storage">
  <SegmentedMeterRoot decorative max={256} size="large">
    <SegmentedMeterItem value={43.3} />
    <SegmentedMeterItem value={28.3} />
    <SegmentedMeterItem value={20.2} />
    <SegmentedMeterItem value={14.9} />
  </SegmentedMeterRoot>
  <Legend
    items={[
      { children: "Videos", slotEnd: "43.3 GB" },
      { children: "Images", slotEnd: "28.3 GB" },
      { children: "Documents", slotEnd: "20.2 GB" },
      { children: "Audio files", slotEnd: "14.9 GB" },
    ]}
  />
</FormField>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **aria-label** * | `string` | - | Defines a string value that labels the current element. Names the meter, exposing it as one image with a text alternative. Use it when the colors are not telling categories apart, so a legend would have nothing to explain. |
| **items** * | `SegmentedMeterItemProps[]` | - | Bars drawn in order along the track, each spread onto a {@link SegmentedMeterItem}. |
| aria-labelledby | `string` | - | Identifies the element (or elements) that labels the current element. Names the meter from an element that already says what it is. Alternative to `aria-label`. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| decorative | `true` | - | Hides the bars from assistive technology, for a meter whose {@link Legend } carries the breakdown. Pass this instead of a name, never alongside one. |
| max | `number` | - | Upper bound of the scale, in the same units as each item's `value`. Whatever the items do not account for renders as empty track.  Leave it unset to measure the items against their own total, filling the track completely. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Height of the track. |

`* = required prop`

## Exports

- Components: `SegmentedMeter`, `SegmentedMeterItem`, `SegmentedMeterRoot`
- Types: `SegmentedMeterAriaProps`, `SegmentedMeterItemProps`, `SegmentedMeterProps`, `SegmentedMeterRootProps`, `SegmentedMeterRootPropsBase`, `SegmentedMeterSize`
