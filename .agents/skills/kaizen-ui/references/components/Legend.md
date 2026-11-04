# Legend

A key mapping colors to the data series they represent.

## Notes

- A legend is what makes a color-coded visualization accessible — without it, category is conveyed by hue alone. Render one whenever more than one color is in play.

- Set an entry's `color` to override the categorical default. It takes any CSS color value and is applied as-is, so a design token reference, your own custom property, or a literal all work.

- Pin `color` on both sides whenever a legend describes something whose order can change, so the two cannot disagree.

## Examples

### Basic Legend

Entries fall back to the categorical data colors in order, so a legend built from `items` lines up with a chart or meter that does the same without any color wiring. In cases where the Legend can get wide, you should cap the width at `max-w-80` and horizontally center it `mx-auto`.

```tsx
<Legend
  className="mx-auto max-w-80"
  items={[
    { children: "Videos", slotEnd: "433.27 GB" },
    { children: "Images", slotEnd: "28.32 GB" },
    { children: "Documents", slotEnd: "56.66 MB" },
    { children: "Audio files", slotEnd: "4.86 MB" },
  ]}
/>
```

### Horizontal Legend

Use the horizontal orientation for a legend that sits beside or beneath a chart, where each entry is a label rather than a label and value pair.

```tsx
<Legend items={[{ children: "Videos" }, { children: "Images" }, { children: "Documents" }]} orientation="horizontal" />
```

### Custom Colors Legend

`color` is an ordinary CSS color value applied straight to the swatch. Prefer the `--color-chart-data-categorical-*` tokens where you can, since they follow the active light or dark theme.

```tsx
<Legend
  items={[
    { children: "Literal", color: "#c1440e" },
    { children: "Token", color: "var(--color-chart-data-categorical-6)" },
  ]}
/>
```

### Composed Legend

Compose entries directly to pin each category to a specific color, or to render richer content than a label and value.

```tsx
<LegendRoot>
  <LegendItem color="var(--color-chart-data-categorical-1)" slotEnd="58 nodes">
    Healthy
  </LegendItem>
  <LegendItem color="var(--color-chart-data-categorical-3)" slotEnd="24 nodes">
    Degraded
  </LegendItem>
  <LegendItem color="var(--color-chart-data-categorical-7)" slotEnd="12 nodes">
    Critical
  </LegendItem>
</LegendRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `LegendItemProps[]` | - | Entries rendered in order. Each is spread onto a {@link LegendItem}. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| orientation | `"horizontal" \| "vertical"` | `"vertical"` | How the entries flow. - `"vertical"` - Stacks entries, label on the left and value on the right. Suited to a legend beneath a meter or a value breakdown. - `"horizontal"` - Lays entries out in a wrapping row. Suited to a legend beside or beneath a chart. |

`* = required prop`

## Exports

- Components: `Legend`, `LegendItem`, `LegendRoot`
- Types: `LegendItemProps`, `LegendProps`, `LegendRootProps`
