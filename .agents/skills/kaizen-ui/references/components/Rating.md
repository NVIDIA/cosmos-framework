# Rating

A row of icons that shows a value within a range, or lets the user pick one.

## Notes

- Pass `readOnly` to display a value such as an average score. It renders a native `<meter>` and accepts any fraction (`value={3.7}`); without it the rating is a radio group for picking a value.

- Always name the rating: `aria-label`, or wrap it in a `FormField`. In input mode the name labels the radio group; in display mode it labels the meter.

- In input mode, wrap it in `<FormField labelable={false}>`: the rating renders a `<fieldset>`, which a `<label>` cannot point at. Display mode renders a labelable `<meter>` and needs no flag.

- `children` renders as visible text beside the icons, such as `3 / 5`. The rating generates no text of its own.

- Use `step={0.5}` to let users pick half values. Display mode shows any fraction regardless of `step`.

- There is no built-in clear. Control `value` and pair the rating with a `Button` that sets it to `null`.

- Pass `name` to submit the value with a form, exactly like a radio group. `required` blocks submission until a value is picked.

- `aria-valuetext` applies only when `readOnly`; it defaults to "3.5 out of 5" and can be localized.

- Replace the star with `slotIcon` (empty shape) and `slotIconFilled` (filled shape). Both must paint with `currentColor` and share an outline, or partial fills look misaligned.

- Change the fill color by setting the `--nv-rating-fill-color` custom property. Disabled ratings ignore it.

## Examples

### Display Rating

`readOnly` displays a value, such as an average score. It renders a native `<meter>` and keeps any fraction, so 3.7 fills seven tenths of the fourth star. The children are visible text beside the stars; the rating generates none of its own.

```tsx
<Rating readOnly value={3.7} aria-label="Average rating">
  3.7 / 5
</Rating>
```

### Clearable Rating

Without `readOnly` the rating is a native radio group, so it submits under `name` and a form reset restores its default, which with no `defaultValue` is empty. `step={0.5}` splits every star into two halves, each its own radio. A radio cannot be unchecked by clicking it again and there is no built-in clear; outside a form, control `value` and set it to `null`.

```tsx
<form>
  <Flex align="center" gap="3">
    <Rating name="score" step={0.5} aria-label="Rate this model" />
    <Button kind="tertiary" size="small" type="reset">
      Clear
    </Button>
  </Flex>
</form>
```

### Custom Icon Rating

`slotIcon` replaces the empty shape and `slotIconFilled` the filled one. Both must paint with `currentColor`, and they should share an outline so a partial fill lines up. Rename the radios with `getItemLabel` to match the icon.

```tsx
<Rating
  defaultValue={4}
  aria-label="How much do you like it?"
  getItemLabel={(value) => (value === 1 ? "1 heart" : `${value} hearts`)}
  slotIcon={
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M8 14.2 1.9 8.1a3.7 3.7 0 0 1 5.3-5.3L8 3.6l.8-.8a3.7 3.7 0 0 1 5.3 5.3L8 14.2Zm0-1.6 5.3-5.2a2.6 2.6 0 0 0-3.7-3.8L8 5.2 6.4 3.6a2.6 2.6 0 0 0-3.7 3.8L8 12.6Z"
      />
    </svg>
  }
  slotIconFilled={
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path fill="currentColor" d="M8 14.2 1.9 8.1a3.7 3.7 0 0 1 5.3-5.3L8 3.6l.8-.8a3.7 3.7 0 0 1 5.3 5.3L8 14.2Z" />
    </svg>
  }
/>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| defaultValue | `number` | `null` | Initial value when uncontrolled. |
| disabled | `boolean` | - | Renders the disabled colors and disables every radio. |
| getItemLabel | `(value: number) => string` | `(value) => "1 star" | `${value} stars`` | Accessible name for the radio that selects each value. Override it to localize the default. |
| max | `number` | `5` | Number of items, and the highest value. |
| name | `string` | - | Shared HTML `name` for the radios, and the key the value submits under. Falls back to the surrounding `FormField`'s `name`, then to a generated one so the radios still group; inside a form, that generated key submits too. |
| onValueChange | `(value: number) => void` | - | Called with the value the reader picks. Input mode only. |
| readOnly | `boolean` | `false` | Renders a display of the value instead of radios. Accepts any fraction, clamped to `0..max`. |
| required | `boolean` | - | Refuses form submission until a value is picked. |
| slotIcon | `ReactNode` | - | Empty shape for every item. Must paint with `currentColor`. Defaults to the star. |
| slotIconFilled | `ReactNode` | - | Filled shape for every item. Falls back to `slotIcon`, then to the filled star. |
| step | `1 \| 0.5` | `1` | Granularity of the values a reader can pick. `0.5` splits every item into two halves, each its own radio. |
| value | `number` | - | Controlled value. `null` clears the rating. |

## Exports

- Components: `Rating`, `RatingItem`, `RatingLabel`, `RatingRoot`
- Hooks: `useRatingContext`
- Contexts: `RatingContext`
- Types: `RatingContextType`, `RatingItemProps`, `RatingLabelProps`, `RatingProps`, `RatingRootProps`, `RatingStep`

## See also

- [Meter](./Meter.md)
- [RadioGroup](./RadioGroup.md)
