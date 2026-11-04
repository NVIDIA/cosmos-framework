# FormField

Also known as: Form Group, Fieldset

Groups a label, input, and helper text to keep form elements consistent and accessible.

## Notes

- Wrap every form input that needs a visible label, helper text, or validation message. Skip FormField for inputs that use `aria-label` (e.g. toolbar search) and don't need helper text.

- Always provide `slotLabel`. A FormField without a label is inaccessible; if the input doesn't need a visible label, it doesn't need FormField.

- Use one `labelPosition` for every field in a form — mixing top and left labels breaks scanning.

- Pass `labelable={false}` when the field wraps something HTML cannot label — a chart, a group of checkboxes, a `SegmentedMeter`. The label then drops its `htmlFor`.

- `kind` (flat vs. floating) is a form-level decision — pick one and apply it to every input in the form. Note that `kind="floating"` goes on the child input, not on FormField. Mixing flat and floating creates inconsistent label placement and hurts scanning.

- Validate individual fields on blur (set `status="error"` with `slotError`), not on every keystroke. Reserve `status="success"` for fields that need explicit confirmation (e.g. availability checks).

- When `status="error"`, the error message replaces help text in the same space — do not show both simultaneously; it dilutes the error signal.

- Run cross-field validations (e.g. "end date must be after start date") on submit and scroll to the first error field.

- Use `slotHelp` for guidance every user needs (format, constraints); use `slotInfo` for supplementary context that most users can ignore.

- Keep `slotHelp` concise and single-line; push longer guidance to `slotInfo`, a linked doc, or a separate panel.

- Anti-pattern for `slotInfo`: if every user needs the info, it should be `slotHelp`, not hidden behind a popover.

- Switch and Checkbox have built-in labels, so FormField is only needed when those fields require helper text, validation, or an info popover.

## Examples

### Basic Form Field

```tsx
<FormField slotLabel="Email Address" slotHelp="We will not share your email.">
  {/* can contain any KUI form control - TextInput, TextArea, Select, Switch, Checkbox, Upload */}
  <TextInput type="email" pattern="^[^\s@]+@[^\s@]+\.[^\s@]+$" placeholder="name@example.com" required />
</FormField>
```

### Upload Form Field

Upload needs renderInput when used in a FormField to ensure uploaded items render below the FormField wrapper.

```tsx
<Upload
  renderInput={(inputNode) => (
    <FormField name="documents" slotLabel="Upload Documents" slotHelp="Files will be deleted after 30 days.">
      {inputNode}
    </FormField>
  )}
  required
/>
```

### Form Field Status

`status` determines helper text. No status means `slotHelp` shown, `error` means `slotError` shown, and `success` means `slotSuccess` shown.

```tsx
<FormField slotLabel="Email Address" slotError="Please enter a valid email address." slotSuccess="Email address is valid." slotHelp="We will not share your email." status="error">
  <TextInput placeholder="name@example.com" />
</FormField>
```

### Required Form Field

When set, renders an asterisk next to the label. If the wrapped input has `required` set, the form field will automatically render the required indicator(preferred). You can also set it on the FormField itself.

```tsx
<Flex>
  <FormField slotLabel="Email Address">
    <TextInput
      type="email"
      // FormField automatically infers required from the input(preferred)
      required
      placeholder="name@example.com"
    />
  </FormField>
  <FormField
    slotLabel="Full Name"
    // sets required on KUI form controls rendered within
    required
  >
    <TextInput placeholder="Jane Doe" />
  </FormField>
</Flex>
```

### With Info Form Field

Use when the field label alone is insufficient and users benefit from an info popover.

```tsx
<FormField slotLabel="API Key" slotInfo="Your API key is used to authenticate requests." slotHelp="Found in your account settings.">
  <TextInput placeholder="sk-..." />
</FormField>
```

### With Label End Form Field

Use `slotLabelEnd` for brief context that belongs on the label row, such as an optional indicator or a meter reading. It is not automatically included in the control's accessible name or description.

```tsx
<FormField slotLabel="Project name" slotLabelEnd={<span className="ml-auto">Optional</span>}>
  <TextInput placeholder="DGX Cloud deployment" />
</FormField>
```

### Left Label Form Field

Top-aligned labels(default) enhance clarity, provide space for longer labels, and improve scannability on various screen sizes. Left-aligned labels should be used for dense forms to save space or for multi-column layouts. Note that left aligned labels will automatically switch to vertical layout if there is insufficient space.

```tsx
<FormField labelPosition="left" slotLabel="Username" slotHelp="Choose a unique username.">
  <TextInput placeholder="jdoe" />
</FormField>
```

### Composed

Use composed primitives when you need full control over the form field layout. `FormFieldRoot` derives the accessibility context (ids and ARIA wiring) from its props, so `FormFieldLabel`, `FormFieldHelper`, and KUI inputs connect automatically without manual id management.

```tsx
<FormFieldRoot id="demo-field" name="demo-field" status="error" required>
  <FormFieldContentGroup>
    <FormFieldLabelGroup>
      <FormFieldLabel>Label</FormFieldLabel>
    </FormFieldLabelGroup>
    <TextInput placeholder="Composed field" />
  </FormFieldContentGroup>
  <FormFieldHelper>Helper text for the field.</FormFieldHelper>
</FormFieldRoot>
```

### Custom Form Field

Use FormField primitives to create custom form field layouts with automatic ARIA wiring. Here is an example of a row for a settings page.

```tsx
<FormFieldRoot name="theme">
  <Flex paddingY="density-2xl" gap="density-2xl" justify="between" align="center">
    <Flex direction="col" gap="density-xs">
      <FormFieldLabel>
        <Text kind="body/regular/md">Theme</Text>
      </FormFieldLabel>
      <FormFieldHelper>
        <Text kind="body/regular/md">Applies immediately across the workspace.</Text>
      </FormFieldHelper>
    </Flex>
    <Select className="max-w-[240px]" items={["System", "Light", "Dark"]} placeholder="System" />
  </Flex>
</FormFieldRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ Label?: NativeElementAttributes<"label", typeof Label>; FormFieldLabelGroup?: NativeElementAttributes<"div", typeof FormFieldLabelGroup>; FormFieldContentGroup?: NativeElementAttributes<"div", typeof FormFieldContentGroup>; FormFieldControlGroup?: NativeElementAttributes<"div", typeof FormFieldControlGroup>; FormFieldHelper?: NativeElementAttributes<"div", typeof FormFieldHelper>; } & FormFieldInfoProps["attributes"]` | - | Native HTML attributes forwarded to the internal composed components. |
| children | `string \| number \| bigint \| React.ReactElement<unknown, string \| React.JSXElementConstructor<any>> \| Iterable<React.ReactNode> \| React.ReactPortal \| Promise<AwaitedReactNode> \| ((args: FormFieldContextType) => React.ReactElement<unknown, string \| React.JSXElementConstructor<any>>) \| boolean` | - | Input element(s) wrapped by the field, or a render function that receives the field's accessibility context. |
| id | `string` | - | Unique identifier used to associate the label and helper text with the underlying input. Falls back to an auto-generated ID. |
| labelable | `boolean` | `true` | Whether the field wraps a labelable element. Set it `false` around something HTML cannot label — a chart, a group of checkboxes, a `SegmentedMeter` — and the label drops its `htmlFor`. |
| labelPosition | `"left" \| "top"` | `"top"` | Placement of the label relative to the input. - "top" - Standard creation flows and most forms. Supports variable-width inputs and is the fastest position for completion. - "left" - Dense settings panels or key-value layouts with short, consistent labels and constrained vertical space. |
| name | `string` | - | Form control name submitted with the field's value. |
| required | `boolean` | - | When true, indicates that the user is required to fill out this field. Used to determine whether or not the asterisk is shown next to the label.  By default, this will automatically determine if the form field contains a `:required` input, otherwise it will be manually controlled. |
| slotError | `ReactNode` | - | Message rendered below the input when `status` is `"error"`. |
| slotHelp | `ReactNode` | - | Helper message rendered below the input when no validation status is set. |
| slotInfo | `ReactNode` | - | Content shown inside the popover triggered by the info icon next to the label. |
| slotLabel | `ReactNode` | - | Label content rendered above or beside the input. |
| slotLabelEnd | `ReactNode` | - | Content shown after the label. |
| slotSuccess | `ReactNode` | - | Message rendered below the input when `status` is `"success"`. |
| status | `"error" \| "success"` | - | Validation state that drives styling, selects the helper message, and marks an errored control as invalid. |

## Exports

- Components: `FormField`, `FormFieldContentGroup`, `FormFieldControlGroup`, `FormFieldHelper`, `FormFieldInfo`, `FormFieldLabel`, `FormFieldLabelGroup`, `FormFieldRoot`
- Types: `FormFieldContentGroupProps`, `FormFieldControlGroupProps`, `FormFieldHelperProps`, `FormFieldInfoProps`, `FormFieldLabelGroupProps`, `FormFieldLabelProps`, `FormFieldOrientation`, `FormFieldProps`, `FormFieldRootProps`
