# ToggleButton

Also known as: Toggle, Pressed Button, Stateful Button

A button that switches an option on or off and keeps showing which state it is in. Use for a state that persists after the click, such as pinning, starring, or bold text.

## Notes

- Use ToggleButton for an on/off state that acts on nearby content and persists after the click (Pin, Star, Bold, Mute). Use Switch for a labeled setting or preference such as notifications; Checkbox for selections submitted with a form, since a toggle button submits no value; SegmentedControl when exactly one view or mode must always be selected; Button or ButtonGroup for actions that do not leave a state behind.

- Pass the icon to `slotIcon`, not `children`, so it switches from its line to its fill variant while pressed. Do not set `variant` on it unless the icon must stay the same in both states.

- Keep the accessible name constant across states, because screen readers announce the pressed state separately. Never change `children` or `aria-label` to describe the current state; to show different visible text while pressed, pass it to `slotPressed` ("Pin" becomes "Pinned"), and never repeat the `slotIcon` icon there. To swap to a different glyph, put the two icons in `children` and `slotPressed` instead of `slotIcon`.

- Icon-only toggle buttons must set `aria-label` to the name of the option ("Bold", "Pin"), not the action or the state ("Make bold", "Pinned").

- Group related toggles (text formatting, view options) in a ToggleButtonGroup rather than placing several standalone ToggleButtons side by side.

- Default to `color="neutral"`. Reserve `color="brand"` for the most prominent toggle or toggle group in a view.

## Examples

### Basic Toggle Button

Pass the icon to `slotIcon`. It shows its line variant while unpressed and its fill variant while pressed.

```tsx
<ToggleButton slotIcon={<Clock />}>Timestamps</ToggleButton>
```

### Pressed Label Toggle Button

Pass `slotPressed` when the visible label changes with the state. The accessible name stays "Pin" in both states while assistive technology announces the pressed state separately, so never swap `children` or `aria-label` yourself. The `slotIcon` icon stays in place, so leave it out of `slotPressed`.

```tsx
<ToggleButton slotIcon={<Pin />} slotPressed="Pinned">
  Pin
</ToggleButton>
```

### Icon Only Toggle Button

Icon-only toggle buttons need an `aria-label` that names the option ("Star"), not the action or the current state.

```tsx
<ToggleButton aria-label="Star" slotIcon={<Star />} />
```

### Brand Toggle Button

Use `color="brand"` for the one toggle that should stand out in a view, such as the reasoning mode in a chat composer. Unpressed, it matches every other toggle button.

```tsx
<ToggleButton color="brand" defaultPressed>
  Reasoning
</ToggleButton>
```

### Controlled Toggle Button

Control `pressed` when the state lives outside the button, such as a job subscription saved to the server.

```tsx
() => {
  const [watching, setWatching] = useState(false);
  return (
    <ToggleButton pressed={watching} onPressedChange={setWatching} slotIcon={<Bell />}>
      Watch
    </ToggleButton>
  );
};
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| color | `"brand" \| "neutral"` | `"neutral"` | The fill shown while the button is pressed. Unpressed toggle buttons share one outlined look in every color. |
| defaultPressed | `boolean` | `false` | Whether the button is pressed when it first renders. Use when the pressed state is uncontrolled. |
| disabled | `boolean` | - | Disables the button. Inside a group, this takes precedence over the group's `disabled`. |
| onPressedChange | `(pressed: boolean) => void` | - | Called with the next pressed state when the user toggles the button. |
| pressed | `boolean` | - | Controlled pressed state. Pair with `onPressedChange` to manage state externally. |
| size | `"small" \| "tiny" \| "medium" \| "large"` | `"medium"` | The size of the button. |
| slotIcon | `React.ReactElement<{ variant?: "line" \| "fill"; }, string \| React.JSXElementConstructor<any>>` | - | Icon shown before the label. It renders its line variant while unpressed and its fill variant while pressed, unless it sets `variant` itself. Decorative: the accessible name still comes from `children` or `aria-label`. |
| slotPressed | `ReactNode` | - | Content shown in place of `children` while the button is pressed, such as "Pinned" for a "Pin" button. The accessible name still comes from `children`, so it stays the same in both states. The `slotIcon` icon stays in place, so leave it out of this content. |
| value | `string` | - | Identifies the button inside a toggle button group, whose value lists the pressed buttons. |

## Exports

- Components: `ToggleButton`
- Types: `ToggleButtonProps`

## See also

- [ToggleButtonGroup](./ToggleButtonGroup.md)
- [Switch](./Switch.md)
- [SegmentedControl](./SegmentedControl.md)
