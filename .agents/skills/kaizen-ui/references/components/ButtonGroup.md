# ButtonGroup

A row of related buttons presented as a single set of actions with consistent styling.

## Notes

- Cap a ButtonGroup at four items. If more actions are needed, move overflow into a Dropdown rather than widening the group.

- Prefer kind="secondary" or "tertiary" on the group. Multiple kind="primary" buttons in one group compete visually and dilute the page's primary action.

- Only group actions that operate on the same entity or context. Independent or unrelated actions should be rendered as separate Buttons.

- Note that SegmentedControl communicates "pick one" (mutually exclusive); ButtonGroup communicates "here are related actions"; individual Buttons are for independent actions.

- Pick `groupKind` by spacing intent: "gap" for clear visual separation (default for primary), "flush" for tightly coupled actions like split buttons, "border" for a subtle separator without full gap spacing.

- Action overflow / split button pattern: pair a primary Button with a Dropdown for alternatives — the primary action stays prominent while alternatives remain discoverable.

## Examples

### Basic Button Group

Use when offering a set of related choices. Recommended to show no more than 4 actions.

```tsx
<ButtonGroup items={[{ children: "Start Training" }, { children: "Validate" }, { children: "Stop" }]} />
```

### With Children Button Group

Use the children API when you need more control over the button group, such as adding dropdowns. Buttons rendered within will use the ButtonGroup's size/kind/color.

```tsx
<ButtonGroup size="large" color="brand">
  <Button>Start Training</Button>
  <Button>Stop</Button>
  <Dropdown
    // no children, so Dropdown renders as chevron only icon so aria-label is required
    aria-label="Validate or See Details"
    items={["Validate", "See Details"]}
  />
</ButtonGroup>
```

### With Icon Button Group

```tsx
<ButtonGroup items={[{ children: "Start Training" }, { children: "Validate" }, { children: <SettingsIcon />, key: "settings", "aria-label": "Settings" }]} />
```

### Disabled Button Group

Use when the entire group of actions is unavailable due to a precondition.

```tsx
<ButtonGroup disabled items={[{ children: "Start Training" }, { children: "Validate" }, { children: "Stop" }]} />
```

### Disabled Individual Button

You can also disable individual buttons in the group by passing a `disabled` prop to the button. We recommend using children in this case so you can provide a tooltip explaining why the button is disabled.

```tsx
<ButtonGroup>
  <Tooltip disabled={!trainingDisabled} slotContent="Queue is full">
    <Button disabled={trainingDisabled}>Train Model</Button>
  </Tooltip>
  <Button color="danger">Delete Dataset</Button>
</ButtonGroup>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| color | `"brand" \| "neutral" \| "danger"` | `"neutral"` | The color variant of the button. - "brand" - Prominent calls-to-action such as page-level CTAs and modal primary actions. - "neutral" - Regular actions, suitable for most use cases. - "danger" - Destructive actions that cannot be undone, such as delete or remove. |
| disabled | `boolean` | - | Disables every button in the group. |
| groupKind | `"gap" \| "flush" \| "border"` | - | Overrides the divider style between buttons. Defaults to "gap" for primary, "flush" for secondary, and "border" for tertiary. |
| items | `ButtonGroupItem[]` | - | Buttons rendered from a configuration array. Mutually exclusive with `children`. |
| kind | `"primary" \| "secondary" \| "tertiary"` | `"primary"` | The kind of button. - "primary" - The most important call-to-action on the page. Only one per context. - "secondary" - Regular actions, suitable for most use cases. - "tertiary" - Low-priority or supplemental actions. |
| size | `"small" \| "tiny" \| "medium" \| "large"` | `"medium"` | The size of the button. - "large" - The main call-to-action for a page or section. - "medium" - Suitable for most use cases. - "small" - Compact layouts with limited space or less significant actions. - "tiny" - Dense layouts such as table cells where horizontal space is at a premium. |

## Exports

- Components: `ButtonGroup`
- Types: `ButtonGroupProps`

## See also

- [Button](./Button.md)
- [SegmentedControl](./SegmentedControl.md)
- [Dropdown](./Dropdown.md)
