# Label

A text label for a form control. Prefer FormField for end-user forms; this primitive is intended for composing custom field layouts.

## Notes

- The required indicator belongs on the label that names the field, and only once. Inside a `FormField`, set `required` there instead — it already renders the indicator on the field label.

- `required` on Label is presentational. Assistive tech hears required state from the control, so set `required` (or `aria-required`) there too.

## Examples

### Basic Label

Use when labeling a form input to provide accessible context for the control.

```tsx
<Label htmlFor="email">Email Address</Label>
```

### Disabled Label

Use when the associated input is disabled to visually communicate the inactive state.

```tsx
<Label htmlFor="email" disabled aria-disabled="true">
  Email Address
</Label>
```

### Required Label

Renders the required indicator after the label text. It is presentational, so set `required` on the control as well — that is what assistive tech announces.

```tsx
<Flex direction="col" gap="2">
  <Label htmlFor="required-email" required>
    Email Address
  </Label>
  <TextInput id="required-email" name="email" placeholder="name@example.com" required type="email" />
</Flex>
```

### Small Label

Use in compact layouts or alongside small-sized inputs where space is limited.

```tsx
<Label htmlFor="email" size="small">
  Email Address
</Label>
```

### Label With Icon

Use when the label needs an inline help affordance. Label accepts arbitrary children so an icon or tooltip can sit alongside the text.

```tsx
<Label htmlFor="email">
  Email Address
  <Tooltip slotContent="We use this to send account confirmations.">
    <InfoCircle />
  </Tooltip>
</Label>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| disabled | `boolean` | - | Styles the label as disabled, for use only with disabled form controls. |
| htmlFor | `string` | - | ID of the form control this label is associated with. |
| required | `boolean` | - | Renders the required indicator after the label text. Presentational only — also set `required` (or `aria-required`) on the control so assistive tech announces it. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Typographic size of the label. |

## Exports

- Components: `Label`
- Types: `LabelProps`
