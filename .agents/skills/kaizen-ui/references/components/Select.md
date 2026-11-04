# Select

An input that lets the user pick one or more values from a predefined list.

## Notes

- Use Select for bounded, predefined option lists. Below five options, showing everything inline (RadioGroup or Checkbox group) reduces cognitive load because the user doesn't need to remember what's hidden behind the dropdown.

- Reach for Combobox when the list is long enough that filtering helps, or when the option count is unknown or could grow at runtime. The Select/Combobox boundary isn't a hard count — favor Combobox for many options, Select for a small bounded set.

- Select is for picking values, not triggering actions — use Dropdown for action menus.

- Write descriptive placeholders ("Select a region", "Choose a GPU type") instead of generic "Select..." text.

- Pre-select a default only when one option is clearly the common choice; an arbitrary default implies a recommendation that doesn't exist. For required fields with a clearly common choice, pre-selecting removes one interaction step for the common case.

- Group items only when there are 10+ options with natural categories; keep groups to 3–5. Alphabetical ordering optimizes for lookup, not selection — order groups by usage frequency when selection is the primary task.

- Enable `dismissible` on optional fields and leave it off on required fields (a required field must always have a value).

- Single-select closes the listbox on selection; multi-select keeps it open and accumulates picks.

- For multi-select, set `multipleMode="tags"` so each selection shows as a removable tag; the default `"count"` shows a compact summary and becomes `"tags"` in the next major release. Use `renderValue` for a custom display.

- Top-level props are split by element kind: recognized `<button>` attributes go to `SelectToggle`; every other top-level prop (`className`, `onPointerEnter`, `data-*`, …) is spread onto `SelectTrigger`'s `InputShell` (the field chrome). Use `attributes.SelectContent` when something must target the popover.

- Default to `size="medium"` to match NVIDIA's uniform control-height rule. Use `size="small"` only when an explicit dense-layout requirement calls for it.

## Examples

### Basic Select

Use object items for more control over the displayed label and value. String items are convenient for simple cases.

```tsx
<Flex direction="col" gap="density-md">
  <Select items={["Apple", "Banana", "Cherry"]} defaultValue="Banana" placeholder="Choose a fruit" />
  <Select items={["Apple", { children: "Banana", value: "b" }, "Cherry"]} defaultValue={["Apple", "b"]} multiple placeholder="Choose multiple fruits" />
</Flex>
```

### Rich Item Content Select

Pass `slotStart` / `slotEnd` on an item for leading or trailing icons, or use `children` directly for richer multi-line content. Set `filterValue` whenever `children` is non-string so type-ahead still works. Use slotStart and slotEnd on the Select itself to add leading/trailing icons.

```tsx
<Select
  items={[
    {
      value: "document",
      children: "Document",
      slotStart: <Document />,
    },
    {
      value: "schedule",
      children: "Schedule",
      slotStart: <Calendar />,
    },
    {
      value: "report-q1",
      filterValue: "Q1 Report.pdf",
      children: (
        <Flex direction="col">
          <Flex gap="1" align="center">
            <DocumentLine />
            Q1 Report.pdf
          </Flex>
          <Text kind="label/regular/sm">Updated 2 days ago</Text>
        </Flex>
      ),
    },
  ]}
  placeholder="Choose an option"
  slotStart={<Filter />}
  slotEnd={<InfoCircle />}
/>
```

### Custom Render Value Select

Use renderValue when the trigger needs richer or differently formatted content than the matching item's children.

```tsx
<Select items={["Apple", "Banana", "Cherry"]} defaultValue="Apple" renderValue={(value) => (value ? `Selected: ${value}` : "No value selected")} />
```

### Grouped Items Select

Use when options logically belong to categories that help users scan the list.

```tsx
<Select
  items={[
    { slotHeading: "Fruits", items: ["Apple", "Banana"] },
    { slotHeading: "Vegetables", items: ["Carrot", "Lettuce"] },
  ]}
  placeholder="Choose an item"
/>
```

### Multi Tag Select

Set multipleMode="tags" so users can see and remove individual selections. Backspace on the focused trigger removes the last selection.

```tsx
<Select multiple multipleMode="tags" aria-label="GPU models" defaultValue={["H100", "A100"]} items={["H100", "A100", "L40S"]} />
```

### Floating Trigger Select

Use triggerKind='floating' when the Select sits on top of dense content (e.g. a toolbar) and should not render the default input border or background.

```tsx
<Select items={["Apple", "Banana", "Cherry"]} defaultValue="Banana" triggerKind="floating" />
```

### Size Select

Default to `medium` to match NVIDIA's uniform control-height rule. Use `size="small"` only when an explicit dense-layout requirement calls for it. The menu density derives from size by default; override it with the density prop when needed.

```tsx
<Flex direction="col" gap="2">
  <Select size="small" items={["Apple", "Banana"]} placeholder="Small" />
  <Select size="medium" items={["Apple", "Banana"]} placeholder="Medium" />
  <Select size="large" items={["Apple", "Banana"]} placeholder="Large" />
</Flex>
```

### Read Only Select

Use readOnly (instead of disabled) when the value should remain visible and copyable but the user cannot open the menu or change the selection — common in review or summary screens.

```tsx
<Select items={["Apple", "Banana"]} readOnly defaultValue="Apple" />
```

### With Form Field Select

Wrap Select in FormField to attach a label, helper text, and validation status that stay in sync with the trigger via context.

```tsx
<FormField slotLabel="Contact Method" slotHelp="How would you like us to reach you?">
  <Select items={["Phone", "Email", "Both"]} placeholder="Select" />
</FormField>
```

### In Form Select

Pass a name prop so the Select renders a hidden native <select> that participates in standard form submission — works without JavaScript.

```tsx
<form>
  <Flex direction="col" gap="density-md">
    <Select name="fruit" items={["Apple", "Banana", "Cherry"]} placeholder="Pick a fruit" />
    <Button type="submit">Submit</Button>
  </Flex>
</form>
```

### Controlled Select

Use controlled mode when the selection state lives outside the Select — for example, when syncing with form libraries, URL state, or another component.

```tsx
() => {
  const [value, setValue] = useState("Apple");
  return <Select items={["Apple", "Banana", "Cherry"]} value={value} onValueChange={setValue} />;
};
```

### Composed

Use the composed primitives when you need full control over trigger rendering, content layout, or custom items.

```tsx
<SelectRoot defaultValue="b">
  <SelectTrigger placeholder="Pick one" />
  <SelectContent>
    <SelectListbox>
      <SelectItem value="a">Option A</SelectItem>
      <SelectItem value="b">Option B</SelectItem>
      <SelectItem value="c">Option C</SelectItem>
    </SelectListbox>
  </SelectContent>
</SelectRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `(string \| Pick<SelectItemProps, "children" \| "value" \| "filterValue" \| "danger" \| "disabled" \| "onSelect" \| "slotStart" \| "slotEnd"> \| SelectSectionEntry)[]` | - | Options rendered in the listbox. Accepts plain strings, item objects, or section entries that group items under a heading. |
| allowBackspaceRemoval | `boolean` | ``true` with `multipleMode="tags"`, otherwise `false`` | In multi-select mode, allows pressing `Backspace` on the focused trigger to remove the last selected value. On by default when `multipleMode="tags"` shows the selections as tags; otherwise opt in only when the visible selection (e.g. via `renderValue`) makes the target value obvious to the user. |
| attributes | `{ SelectTrigger?: NativeElementAttributes<"button", typeof SelectTrigger>; SelectContent?: NativeElementAttributes<"div", typeof SelectContent>; SelectListbox?: NativeElementAttributes<"menu", typeof SelectListbox> }` | - | Native HTML attributes forwarded to the internal composed components. |
| autoFocusOnHide | `boolean` | `true` | Whether focus returns to the trigger when the listbox closes. |
| defaultOpen | `boolean` | `false` | Whether the listbox is open on first render. Use for uncontrolled open state. |
| defaultValue | `string \| string[] & string \| string[] \| string & string[]` | - | The value of the select when initially rendered. Use when you do not need to control the state of the select.  Use a string for single value selects and an array of strings for multiple value selects. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces. |
| disabled | `boolean` | - | Prevents the user from opening the listbox or changing the value. |
| dismissible | `boolean` | - | Renders a clear button on the trigger that resets the selection. |
| hideOnEscape | `boolean` | `true` | Whether pressing `Escape` closes the listbox. |
| multiple | `boolean` | - | When `mutiple` is true the Select allows for multiple values to be selected. |
| multipleMode | `"count" \| "tags"` | `"count"` | Displays multi-select values as removable tags or a compact count summary. Prefer `"tags"`; the default will change to `"tags"` in the next major release. |
| name | `string` | - | Name applied to the hidden native `<select>` so the value participates in form submission. |
| onKeyDown | `KeyboardEventHandler<HTMLButtonElement>` | - | Keyboard event handler invoked on the trigger before the Select's built-in shortcuts run. Call `event.preventDefault()` to cancel the default ArrowDown/ArrowUp open and (when enabled) Backspace removal behavior. |
| onOpenChange | `(open: boolean) => void` | - | Fires when the listbox opens or closes. |
| onScrollToBottom | `() => void` | - | Called when the menu has been scrolled to the bottom. Use to lazy-load additional items. |
| onValueChange | `(value: string) => void \| (value: string[]) => void` | - | Callback when the selected value of the select changes. |
| open | `boolean` | - | Controlled open state of the listbox. |
| placeholder | `ReactNode` | - | Text shown when no value is selected. |
| portal | `boolean` | `false` | Renders the floating content in a React portal. |
| portalContainer | `HTMLElement` | `document.body` | Container for portaled floating content when no nearer portal target exists. |
| readOnly | `boolean` | - | Locks the current value and prevents the listbox from opening, while keeping the trigger focusable. |
| renderValue | `(value: string, setValue: (value: string \| ((prev: string) => string)) => void) => ReactNode \| (value: string[], setValue: (value: string[] \| ((prev: string[]) => string[])) => void) => ReactNode` | - | Controls the rendering of the selected select value in the trigger.  By default the select will render the children of the selected item for a single value select. For a multiple value select, the select renders the count of selected items. |
| required | `boolean` | - | Marks the Select as required for form validation. |
| selectRef | `Ref<HTMLSelectElement>` | - | Forwarded ref to the hidden native `<select>` used for form integration. Populated only when `name` is set. |
| side | `"right" \| "bottom" \| "left" \| "top"` | `"bottom"` | Preferred side relative to the trigger to position the listbox. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Visual size of the trigger. Drives the listbox density unless `density` is set explicitly on the listbox. |
| slotEnd | `ReactNode` | - | Content rendered after the value, before the dismiss/chevron controls. |
| slotStart | `ReactNode` | - | Content rendered before the value, typically an icon. |
| status | `"error" \| "success"` | - | The status of the input. Use `withValidation` to automatically apply success/error states based on `:user-valid` and `:user-invalid` pseudo classes. |
| tabBehavior | `"close" \| "navigate"` | `"close"` | Whether Tab closes the popup or moves through its enabled items while keeping it open. |
| triggerKind | `"flat" \| "floating"` | `"flat"` | Visual treatment of the trigger. Use `"floating"` to drop the background and border for placement on busy surfaces. |
| value | `string \| string[] & string \| string[] \| string & string[]` | - | The controlled value of the select. Must be used in conjunction with `onValueChange`. |

`* = required prop`

## Exports

- Components: `Select`, `SelectContent`, `SelectItem`, `SelectListbox`, `SelectRoot`, `SelectTrigger`
- Hooks: `useSelectContext`
- Contexts: `SelectContext`
- Types: `SelectContentProps`, `SelectContextValue`, `SelectDefaultItem`, `SelectEntry`, `SelectItemProps`, `SelectListboxProps`, `SelectProps`, `SelectRootProps`, `SelectSectionEntry`, `SelectSingleAndMultipleProps`, `SelectTriggerProps`, `SelectTriggerVariantProps`, `SelectValue`

## See also

- [Combobox](./Combobox.md)
- [RadioGroup](./RadioGroup.md)
- [Dropdown](./Dropdown.md)
- [FormField](./FormField.md)
