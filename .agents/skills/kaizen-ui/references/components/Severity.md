# Severity

A ranked severity level shown as a fixed icon and label pairing. Use it to communicate urgency,
risk, or priority on issues, alerts, and incidents.

## Notes

- Only reach for Severity in products with an established, ranked severity or priority model (incidents, vulnerabilities, support tickets, bug trackers). For one-off status or validation messaging, consider Badge or StatusIndicator instead.

- Override the label via children to match the product's vocabulary (`<Severity level="0">SEV-0</Severity>`), but keep one level-to-wording mapping across the whole app.

- Use `level="none"` rather than omitting the component when a list mixes prioritized and unprioritized rows — a consistent column reads better than a ragged one.

- Use Badge (not Severity) for unranked categorical status like "Active" or "Degraded". Severity is specifically for an ordered scale.

## Examples

### Basic Severity

```tsx
<Severity level={1} />
```

### Severity Levels

The scale runs from `0` (most urgent) down to `5`. Reach for `none` when an item has been triaged but not yet ranked, so a list keeps a consistent column instead of a ragged one.

```tsx
<Flex direction="col" gap="3">
  <Severity level={0} />
  <Severity level={1} />
  <Severity level={2} />
  <Severity level={3} />
  <Severity level={4} />
  <Severity level={5} />
  <Severity level="none" />
</Flex>
```

### Severity Custom Label

`level` is the rank; the label is the wording. Pass children to map a product's own severity vocabulary onto the scale, and keep one level-to-wording mapping across the whole app so it stays comparable between screens.

```tsx
<Flex direction="col" gap="3">
  <Severity level={0}>SEV-0</Severity>
  <Severity level={1}>SEV-1</Severity>
  <Severity level={2}>SEV-2</Severity>
</Flex>
```

### Severity Custom Icon

`slotIcon` is an escape hatch for products whose severity model needs different glyphs. The default pairing is what makes the scale recognizable across apps, so override every level or none of them — sizing and color stay with the component.

```tsx
<Flex direction="col" gap="3">
  <Severity level={0} slotIcon={<Icon name="warning" variant="fill" />}>
    Blocker
  </Severity>
  <Severity level={1} slotIcon={<Icon name="bell" variant="fill" />}>
    Urgent
  </Severity>
</Flex>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| children | `ReactNode` | - | Label rendered beside the icon. Defaults to the level's own name — override it to match a product's own severity vocabulary (`P0`, `SEV-1`, and so on). |
| level | `0 \| "none" \| 1 \| 2 \| 3 \| 4 \| 5` | `"none"` | Position on the severity scale, from `0` (most urgent) down to `5`. Use `none` when no severity has been assigned. A rank outside the scale falls back to `none`, so data that drifts ahead of its types degrades instead of breaking the row. |
| slotIcon | `ReactNode` | - | Icon rendered before the label. Defaults to the level's own icon — the pairing is fixed by design, so only override it when a product's severity model genuinely needs different glyphs. Sizing and color are handled by the component. |

## Exports

- Components: `Severity`
- Types: `SeverityProps`
