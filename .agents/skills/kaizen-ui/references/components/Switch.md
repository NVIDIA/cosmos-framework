# Switch

Also known as: Toggle, ToggleSwitch

A two-state toggle for binary settings whose change takes effect immediately. Commonly used for preferences and feature toggles.

## Notes

- Switch is exclusively for immediate-effect toggles (value takes effect as soon as toggled). If the toggle sits inside a form with a submit button, or the value is only applied on save/submit, use Checkbox instead.

- For terms or consent acknowledgments ("I agree to…"), use Checkbox, not Switch.

- Always provide `slotLabel`, or `aria-label` when the surrounding context already names the setting. An unlabeled Switch is inaccessible.

- Keep `labelSide` consistent across a list of Switches so the controls and labels align in a single column.

- Switch communicates "this change applies now" — unlike Checkbox, which implies "this will be submitted with the form."

## Examples

### Basic Switch

Use when toggling a binary setting that takes effect immediately.

```tsx
<Flex direction="col" gap="density-md">
  <Switch slotLabel="Off" />
  <Switch defaultChecked slotLabel="On" />
</Flex>
```

### Sizes Switch

Use a single size consistently within a group. Default `medium` fits most settings rows; reach for `small` in dense toolbars and `large` for prominent toggles.

```tsx
<div className="flex flex-col items-start gap-4">
  <Switch size="small" slotLabel="Small" />
  <Switch size="medium" slotLabel="Medium" />
  <Switch size="large" slotLabel="Large" />
</div>
```

### Without Label Switch

Use `aria-label` when the switch's purpose is conveyed by surrounding context (e.g. a settings row with its own heading) and a visible label would be redundant.

```tsx
<Switch aria-label="Enable notifications" onCheckedChange={(checked) => console.log(checked)} />
```

### Label Start Switch

Use when aligning labels at the start for consistency in a settings list.

```tsx
<Switch slotLabel="Dark mode" labelSide="start" />
```

### Disabled Switch

Use when a setting is locked by permissions or an external condition. Pair with a tooltip or helper text explaining why.

```tsx
<Tooltip slotContent="Maintenance mode is unavailable while jobs are running">
  <Switch slotLabel="Maintenance mode" disabled defaultChecked />
</Tooltip>
```

### Form Integration Switch

Switch is a native HTML input element defaulting to `type='checkbox'`. Provide `name` and `value` so the switch participates in native form submission.

```tsx
<Switch name="notifications" value="enabled" slotLabel="Enable notifications" />
```

### As Radio Group Switch

```tsx
<Flex direction="col" gap="density-md" role="radiogroup" aria-label="Options">
  <Switch slotLabel="Option 1" type="radio" name="group" />
  <Switch slotLabel="Option 2" type="radio" name="group" />
  <Switch slotLabel="Option 3" type="radio" name="group" />
</Flex>
```

### Composed

Use composed primitives when you need full control over the switch layout or label rendering.

```tsx
<SwitchRoot>
  <SwitchInput aria-label="Enable feature" />
</SwitchRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| attributes | `{ SwitchInput?: NativeElementAttributes<"input", typeof SwitchInput>; Label?: NativeElementAttributes<"label", typeof Label> }` | - | Native HTML attributes forwarded to the internal composed components. |
| labelSide | `"start" \| "end"` | `"end"` | Side of the switch the label sits on. Keep this consistent across a group of switches. |
| onCheckedChange | `(checked: boolean) => void` | - | Fires when the checked state changes. Receives the next checked value. |
| side | `"start" \| "end"` | `"end"` | Side of the switch the label sits on. Keep this consistent across a group of switches. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Visual size of the switch. Keep this consistent across a group of switches. |
| slotLabel | `ReactNode` | - | Label text for the switch. |

## Exports

- Components: `Switch`, `SwitchInput`, `SwitchRoot`
- Types: `SwitchInputProps`, `SwitchProps`, `SwitchRootProps`
