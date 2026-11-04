# Combobox

A text input paired with a filterable listbox for selecting one or more values from a list. Use when users need to search through a large set of options.

## Notes

- Use Combobox when the option list is long enough that filtering helps; reach for Select when the list is short and bounded. The boundary isn't a hard count — favor Combobox when the option count is unknown or could grow at runtime, Select when you know the set stays small. For unconstrained free text use TextInput.

- Multi-select Combobox is preferred over multi-select Select when the option list is long — the filter lets users find items without scrolling through dozens of options.

- Write descriptive placeholders ("Search regions...", "Search team...") so users know they can type to filter.

- Group items only when there are 10+ options with natural categories; keep groups to 3–5 and order them by usage frequency.

- Prefer default `size="medium"` for NVIDIA enterprise contexts; do not pass `size="small"` outside dense-layout exceptions.

- For optional fields, set `dismissible: true` so users can clear the selection. Leave it off on required fields — a required field must always carry a value, so the clear button creates a validation conflict.

- Top-level props are split by element kind: recognized `<input>` attributes (`aria-label`, `placeholder`, `name`, `autoComplete`, …) go to the inner input; every other top-level prop (`onPointerEnter`, `onClick`, `data-*`, …) is spread onto `ComboboxContent` (the popover), not the trigger. Use `attributes.ComboboxTrigger` or `attributes.InputShell` when something must live on the trigger chrome.

- Anti-patterns: Combobox for a handful of options when you know the set stays small (use Select or RadioGroup); missing descriptive placeholder text; Combobox for pure free-text entry.

## Examples

### Basic Combobox

```tsx
<Flex direction="col" gap="2">
  <Combobox aria-label="Fruit" items={["Apple", "Banana", "Cherry", "Date", "Elderberry"]} placeholder="Search fruits..." />
  <Combobox aria-label="Fruit" defaultValue={["Apple", "Banana"]} multiple items={["Apple", "Banana", "Cherry", "Date", "Elderberry"]} placeholder="Search fruits..." />
</Flex>
```

### With Default Value Combobox

```tsx
<Combobox aria-label="Fruit" items={["Apple", "Banana", "Cherry"]} defaultValue="Banana" />
```

### Multiple Combobox

Use when users need to select multiple values with type-ahead filtering. By default, the selected values are rendered as tags in the trigger.

```tsx
<Combobox aria-label="Colors" items={["Red", "Green", "Blue", "Yellow"]} multiple placeholder="Pick colors..." />
```

### Multiple Combobox With Default Values

```tsx
<Combobox aria-label="Colors" items={["Red", "Green", "Blue", "Yellow"]} multiple defaultValue={["Red", "Green"]} placeholder="Pick colors..." />
```

### Combobox With Separate Item Values

```tsx
<Combobox
  aria-label="Colors"
  items={[
    { value: "red", children: "Red (value: red)" },
    { value: "green", children: "Green (value: green)" },
    { value: "blue", children: "Blue (value: blue)" },
    { value: "yellow", children: "Yellow (value: yellow)" },
  ]}
  multiple
  defaultValue={["red", "green"]}
  placeholder="Pick colors..."
/>
```

### Combobox With Rich Item Children

When setting rich children in `items`, you should also provide a `filterValue` which is a text only representation of `children` that will be used for type-ahead filtering.

```tsx
<Combobox
  aria-label="Colors"
  items={[
    {
      value: "red",
      filterValue: "Red",
      children: (
        <Flex align="center" gap="density-md">
          <CircleFill /> Red
        </Flex>
      ),
    },
  ]}
  placeholder="Pick colors..."
/>
```

### Multiple Count Mode Combobox

Use count mode when many selections are expected and individual tags would overflow the trigger.

```tsx
<Combobox aria-label="Tags" items={VERY_LONG_LIST} multiple multipleMode="count" placeholder="Select frameworks..." />
```

### Grouped Items Combobox

Use when options logically belong to categories that help users scan the list.

```tsx
<Combobox
  aria-label="Team member"
  items={[
    { slotHeading: "Engineering", items: ["Ada", "Grace"] },
    { slotHeading: "Design", items: ["Charles", "Dorothy"] },
  ]}
  placeholder="Search team..."
/>
```

### Custom Empty State Combobox

```tsx
<Combobox aria-label="Fruit" items={[]} emptyStateMessage="No fruits found" />
```

### Dismissible Combobox

Use when users need a quick way to clear their selection.

```tsx
<Combobox aria-label="Fruit" items={["Apple", "Banana", "Cherry"]} dismissible defaultValue="Apple" />
```

### Disabled Combobox

Use when the selection is locked by external conditions.

```tsx
<Combobox aria-label="Fruit" items={["Apple", "Banana"]} disabled defaultValue="Apple" />
```

### With Slot Start And End Combobox

```tsx
<Combobox aria-label="Fruit" items={["Apple", "Banana"]} slotStart={<SomeIcon />} slotEnd={<SomeIcon />} />
```

### With Async Loading Combobox

```tsx
<Combobox
  aria-label="Fruit"
  items={VERY_LONG_LIST}
  onScrollToBottom={() => {
    console.log("Loading more items...");
  }}
/>
```

### Composed

Use the composed primitives when you need full control over trigger rendering, content layout, or custom items.

```tsx
<ComboboxRoot defaultValue="Apple">
  <ComboboxTrigger placeholder="Pick a fruit..." />
  <ComboboxContent>
    <ComboboxListbox>
      <ComboboxItem value="Apple">Apple</ComboboxItem>
      <ComboboxItem value="Banana">Banana</ComboboxItem>
      <ComboboxItem value="Cherry">Cherry</ComboboxItem>
    </ComboboxListbox>
  </ComboboxContent>
</ComboboxRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `(string \| Pick<ComboboxItemProps, "children" \| "danger" \| "disabled" \| "filterValue" \| "onSelect" \| "slotEnd" \| "slotStart" \| "value"> \| ComboboxSectionEntry)[]` | - | Declarative list of options, sections, or string shortcuts shown in the listbox. |
| allowBackspaceRemoval | `boolean` | - | In multi-select mode, pressing `Backspace` while the input is empty removes the last selected value. Defaults to `true` when selections render as tags with the built-in renderer, and `false` otherwise. |
| attributes | `{ ComboboxContent?: ComboboxContentProps; ComboboxListbox?: ComboboxListboxProps; ComboboxTrigger?: ComboboxTriggerProps; InputShell?: InputShellProps }` | - | Native HTML attributes forwarded to the internal composed components. |
| children | `(props: { mode: "count" \| "tags"; removeSelectedValue: (value: string) => void; selectedItems: { label: string; value: string; }[]; summaryLabel: string }) => ReactNode` | - | Custom renderer for the selected values. When provided, `allowBackspaceRemoval` defaults to `false`. |
| className | `string` | - | Class names applied to the field container that wraps the input, not to the `<input>` itself. |
| defaultInputValue | `string` | - | Initial input value when uncontrolled. |
| defaultOpen | `boolean` | `false` | Initial open state when uncontrolled. |
| defaultValue | `string \| string[]` | - | The selected value when the Combobox is initially rendered. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces. |
| disabled | `boolean` | - | Disables interaction with the Combobox. |
| dismissible | `boolean` | - | Renders a dismiss button that clears the Combobox value. |
| emptyStateMessage | `string` | `"No results found"` | Message rendered when no items match the current filter. |
| filteringBehavior | `"after-input" \| "on-open"` | `"after-input"` | Controls when the option list begins filtering. - "after-input" - Shows all options when the popover opens, then filters after the user types. - "on-open" - Filters immediately using the current input value. |
| filterMatchFn | `"disable" \| ((matchTerm: string, value: string) => boolean)` | - | Custom matcher used to decide whether an item is visible for the current search value. Defaults to a case- and diacritic-insensitive substring match. Set to `"disable"` to disable filtering. |
| formatSummaryLabel | `(count: number) => string` | - | Optional formatter for the count summary label used when `mode="count"` and by render-prop consumers. |
| hideOnEscape | `boolean` | `true` | Closes the Combobox when Escape is pressed. |
| id | `string` | - | Id applied to the input element. |
| inputValue | `string` | - | Controlled input value. |
| kind | `"flat" \| "floating"` | `"flat"` | Visual treatment of the shell. `"flat"` has a border and background; `"floating"` is borderless and transparent. |
| multiple | `boolean` | `false` | Enable multi-select. When `true`, `value` / `defaultValue` are arrays of strings and `onValueChange` receives an array; when omitted or `false` (the default), the Combobox is single-select and those props are plain strings. |
| multipleMode | `"count" \| "tags"` | `"tags"` | Controls how selected values are rendered in multi-select mode. - "tags" - Render each selection as an individually labelled, removable tag. - "count" - Render a compact summary of how many values are selected. |
| name | `string` | - | Form field name used by the hidden native `<select>`. |
| onInputValueChange | `(value: string) => void` | - | Called when the input text changes. |
| onOpenChange | `(open: boolean) => void` | - | Called when the popover opens or closes. |
| onScrollToBottom | `() => void` | - | Called when the listbox scrolls to the bottom. Useful for paginating or lazy-loading additional options. |
| onValueChange | `((value: string) => void) \| ((value: string[]) => void)` | - | Callback fired when the selected value changes. |
| open | `boolean` | - | Controlled open state. |
| placeholder | `string` | - | Placeholder text shown before a value is selected or typed. Pass `null` to render no placeholder. |
| portal | `boolean` | `false` | Renders the floating content in a React portal. |
| portalContainer | `HTMLElement` | `document.body` | Container for portaled floating content when no nearer portal target exists. |
| readOnly | `boolean` | - | Prevents the value from being changed while preserving form submission. |
| renderValue | `(value: string \| string[], setValue: ((nextValue: string \| string[]) => void) \| ((nextValueFunc: (prevValue: string \| string[]) => string \| string[]) => void)) => ReactNode` | - | Controls the value rendered inside the trigger when the Combobox is closed. |
| required | `boolean` | - | Requires a value for form submission. |
| resetValueOnBlur | `boolean` | `true` | Restores the input text to the selected option label on blur, or clears unmatched text when nothing is selected. |
| selectRef | `Ref<HTMLSelectElement>` | - | Callback ref for the hidden native `<select>` used for form submission. |
| side | `"right" \| "bottom" \| "left" \| "top"` | `"bottom"` | Side of the trigger the popover anchors to. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Visual size of the Combobox trigger. |
| slotEnd | `ReactNode` | - | Content rendered after the input. |
| slotStart | `ReactNode` | - | Content rendered before the input. |
| status | `"error" \| "success"` | - | The status of the input. Use `withValidation` to automatically apply success/error states based on `:user-valid` and `:user-invalid` pseudo classes. |
| style | `React.CSSProperties` | - | Inline styles applied to the field container that wraps the input, not to the `<input>` itself. |
| value | `string \| string[]` | - | Controlled selected value. |

`* = required prop`

## Exports

- Components: `Combobox`, `ComboboxContent`, `ComboboxItem`, `ComboboxListbox`, `ComboboxRoot`, `ComboboxSelectedTags`, `ComboboxToggle`, `ComboboxTrigger`
- Hooks: `useComboboxContext`
- Contexts: `ComboboxContext`
- Types: `ComboboxContentProps`, `ComboboxContextValue`, `ComboboxDefaultItem`, `ComboboxEntry`, `ComboboxFilteringBehavior`, `ComboboxFilterMatchFn`, `ComboboxItemProps`, `ComboboxListboxProps`, `ComboboxProps`, `ComboboxRootProps`, `ComboboxSectionEntry`, `ComboboxSelectedTagsProps`, `ComboboxSelectedTagsRenderProps`, `ComboboxSelectedValuesDisplay`, `ComboboxSingleAndMultipleProps`, `ComboboxToggleProps`, `ComboboxTriggerProps`, `ComboboxValue`

## See also

- [Select](./Select.md)
- [TextInput](./TextInput.md)
- [FormField](./FormField.md)
