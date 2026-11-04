# Menu

A list of actions, toggles, or selections. Use when the list should always be visible (sidebars, settings panels).

## Notes

- Menu does not manage its own visibility. If the list should open from a button click and dismiss on selection, use Dropdown instead, or Select for form value selection.

- Pick one of three icon modes: no icons (default for text-only menus), selective icons (only for items where the icon adds significant recognition value), or all items have icons (for menus where every action benefits from visual scanning).

- Within a single section, either all items have icons or none — mixing icons inside a section causes a left-alignment offset.

- If you're selectively using icons you can align non-icon items by setting `slotStart={true}`

- Icons supplement text labels and never replace them; don't mix filled and outline variants in the same menu; too many icons creates clutter.

- Group items into sections when there are 8 or more items with clear categories; keep section count to 3-5 and order sections by frequency of use. Fewer items don't benefit from grouping, and too many sections defeat the purpose.

- Keep section headings to 1–3 words — they are scan labels, not descriptions.

- Limit sub-menus to one level of nesting; deeper nesting is hard to navigate. Use sub-menus only for clear parent-child relationships, not to shorten long menus — consider Dropdown sections as an alternative.

- Use divider entries to visually separate unrelated groups of items without adding a section heading. `{ kind: "divider" }`

## Examples

### Basic Menu

```tsx
<Menu
  items={[
    {
      children: "Dashboard",
      // onSelect: handleSelect
    },
    {
      children: "Settings",
      // onSelect: handleSelect
    },
    {
      children: "Help",
      // onSelect: handleSelect
    },
  ]}
/>
```

### With Icons Menu

Use when menu items benefit from leading icons to improve scannability and recognition.

```tsx
<Menu
  items={[
    { children: "Profile", slotStart: <Icon />, onSelect: handleSelect },
    {
      children: "Notifications",
      slotStart: <Icon />,
      onSelect: handleSelect,
    },
    {
      children: "Logout",
      slotStart: <Icon />,
      danger: true,
      onSelect: handleSelect,
    },
  ]}
/>
```

### With Sections Menu

Use when items belong to distinct categories that benefit from labeled group headings.

```tsx
<Menu
  items={[
    {
      slotHeading: "File",
      items: [
        { children: "New", onSelect: handleSelect },
        { children: "Open", onSelect: handleSelect },
        { children: "Save", disabled: true, onSelect: handleSelect },
      ],
    },
    {
      slotHeading: "Edit",
      items: [
        { children: "Undo", onSelect: handleSelect },
        { children: "Redo", onSelect: handleSelect },
        { children: "Delete", danger: true, onSelect: handleSelect },
      ],
    },
  ]}
/>
```

### With State Controls

```tsx
<Menu
  items={[
    {
      slotHeading: "Checkbox Items",
      items: [
        { children: "Auto-save", kind: "checkbox", defaultChecked: true },
        { children: "Spell Check", kind: "checkbox" },
        { children: "Word Wrap", kind: "checkbox" },
      ],
    },
    {
      slotHeading: "Radio Items",
      kind: "radio",
      name: "indent",
      defaultValue: "2",
      items: [
        { children: "2 spaces", value: "2" },
        { children: "4 spaces", value: "4" },
        { children: "8 spaces", value: "8" },
      ],
    },
    {
      slotHeading: "Selected State Items(Radio)",
      kind: "radio",
      name: "theme",
      radioKind: "check",
      defaultValue: "light",
      items: [
        { children: "Light", value: "light" },
        { children: "Dark", value: "dark" },
        { children: "System", value: "system" },
      ],
    },
  ]}
/>
```

### Filterable Menu

Use when the menu has many items and the user benefits from filtering by typing. Enable only for long static or async lists (roughly 10+ items). Pass `data-empty-message` to customize the message shown when no items match.

```tsx
<Menu
  filterable
  data-empty-message="No items match your search"
  items={
    [
      // ...long items
    ]
  }
/>
```

### With Form Actions Menu

Use `formAction` and `formMethod` inside a `<form>` to submit menu actions to different endpoints with no client-side JavaScript required.

```tsx
<form>
  <Menu
    items={[
      {
        children: "Approve",
        formAction: "/api/actions/approve",
        formMethod: "post",
      },
      {
        children: "Reject",
        formAction: "/api/actions/reject",
        formMethod: "post",
      },
      {
        children: "Request Changes",
        formAction: "/api/actions/request-changes",
        formMethod: "post",
      },
    ]}
  />
</form>
```

### Composed

```tsx
<MenuRoot>
  <MenuSection slotHeading="Actions">
    <MenuItem
    // onSelect={handleSelect}
    >
      New File
    </MenuItem>
    <MenuItem
    // onSelect={handleSelect}
    >
      Open File
    </MenuItem>
  </MenuSection>
  <MenuSection slotHeading="Preferences">
    <MenuCheckboxItem defaultChecked>Auto-save</MenuCheckboxItem>
    <MenuRadioGroup name="view" slotHeading="View Mode" defaultValue="grid">
      <MenuRadioGroupItem value="grid">Grid</MenuRadioGroupItem>
      <MenuRadioGroupItem value="list">List</MenuRadioGroupItem>
    </MenuRadioGroup>
  </MenuSection>
</MenuRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `MenuEntry[]` | - | Items rendered in the menu, in order. Strings are shorthand for a default item with that label. |
| attributes | `{ MenuSearch?: NativeElementAttributes<"input", typeof MenuSearch> }` | - | Native HTML attributes forwarded to the internal composed components. |
| defaultFilterValue | `string` | - | Initial value for the search input when the menu is filterable and uncontrolled. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces. |
| filterable | `boolean` | `false` | Renders a search input at the top of the menu and filters items as the user types. |
| filterMatchFn | `"disable" \| ((matchTerm: string, value: string) => boolean)` | - | Custom matcher used to decide whether an item is visible for the current search value. Defaults to a case- and diacritic-insensitive substring match. |
| filterValue | `string` | - | Controlled value for the search input. Pair with `onFilterChange`. |
| onFilterChange | `(value: string) => void` | - | Called with the new search value as the user types in a filterable menu. |
| onItemCheckedChange | `(item: { kind: "checkbox" } & BaseMenuItem, checked: "indeterminate" \| boolean) => void` | - | Called when a checkbox item's checked state changes, receiving the entry and the new state. |
| onItemSelect | `(event: Event, item: { kind?: "default" } & BaseMenuItem \| { kind: "checkbox" } & BaseMenuItem \| { kind?: never } & Omit<BaseMenuItem, "danger"> & Pick<RadioGroupInputProps, "danger" \| "value" \| "required">) => void` | - | Called when any selectable item is activated, receiving the underlying event and the entry. |
| ref | `Ref<HTMLMenuElement>` | - |  |

`* = required prop`

## Exports

- Components: `Menu`, `MenuCheckboxItem`, `MenuHeading`, `MenuItem`, `MenuRadioGroup`, `MenuRadioGroupItem`, `MenuRoot`, `MenuSearch`, `MenuSearchProvider`, `MenuSection`
- Hooks: `useMenuSearchHandlers`, `useMenuSearchIsMatch`
- Contexts: `MenuSearchHandlersContext`, `MenuSearchValueContext`
- Utilities: `menuSectionStyles`
- Constants: `DEFAULT_MATCH_FN`, `FOCUSABLE_MENU_ITEM_SELECTOR`
- Types: `BaseMenuItem`, `MenuCheckboxItemEntry`, `MenuCheckboxItemProps`, `MenuDefaultItemEntry`, `MenuDividerItemEntry`, `MenuEntry`, `MenuHeadingProps`, `MenuItemProps`, `MenuKind`, `MenuProps`, `MenuRadioGroupEntry`, `MenuRadioGroupItemProps`, `MenuRadioGroupProps`, `MenuRadioItemEntry`, `MenuRootProps`, `MenuSearchHandlersContextStore`, `MenuSearchProps`, `MenuSearchProviderProps`, `MenuSearchValueContextStore`, `MenuSectionEntry`, `MenuSectionProps`

## See also

- [Dropdown](./Dropdown.md)
- [Select](./Select.md)
- [Combobox](./Combobox.md)
