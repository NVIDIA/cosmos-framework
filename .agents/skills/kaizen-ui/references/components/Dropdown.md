# Dropdown

A button that opens a menu of actions, links, or selectable options. Use for contextual actions, filters, and overflow menus.

## Notes

- Dropdown is for triggering actions — it is not a value selector.

- Dropdown triggers actions; it does not maintain a stored value. If the user's choice is submitted with a form or persisted to state, use Select (or Combobox for searchable selection) instead.

- Positioning defaults: `side="bottom"` is standard; use `"top"` only when the trigger is near the viewport bottom. `align="start"` aligns to the leading edge.

- Order items by frequency of use, not alphabetically — applies to all dropdowns, including unsectioned ones.

- Pair `danger` items with a confirmation step (Modal or inline) so destructive actions are not committed by a single click.

- Limit sub-menus to one level. If actions need deeper hierarchy, flatten or reach for a different UI pattern.

- Use sections only when there are roughly 6+ items with natural groupings; order items and sections by frequency of use, not alphabetically.

- Radio-in-Dropdown is for mutually exclusive options inside an action menu (e.g. sort direction). It is not a replacement for Select in forms.

- Pass `renderLink` for navigation items so framework routing stays client-side.

- When the dropdown must work from server-rendered HTML or open before hydration, leave `open` unset so `DropdownTrigger` keeps its native `popovertarget`; `onOpenChange` can still observe state changes.

- To control an uncontrolled dropdown programmatically without losing native trigger behavior, attach a ref to `DropdownContent` and call `showPopover()` or `hidePopover()` on the content element.

- `DropdownTrigger` always inject a chevron into the trigger button — including when you use `asChild` to wrap your own `<Button>`. Set `showChevron={false}` on icon-only kebab/overflow triggers

- `DropdownTrigger loading` sets `aria-busy` and `aria-disabled` and blocks menu opening after hydration, including with `asChild`. Custom children must forward props and refs, render their own loading indicator, and guard their own handlers; guard native popover activation before hydration.

- Set `tabBehavior="close"` on every composed `DropdownRoot`. Its default stays `"navigate"` for backward compatibility and changes to `"close"` in the next major version.

- Put icons in `item.slotStart` or `DropdownItem.slotStart` for composed, to ensure icons are aligned correctly with text labels.

- Common mistakes: using Dropdown as a form value selector, nesting submenus 2+ levels deep, exposing a danger action without confirmation, and missing `renderLink` for href items in framework-routed apps, omitting `showChevron={false}` on icon-only triggers, and passing `DropdownItem` icons as children instead of `slotStart`.

## Examples

### Basic Dropdown With Links

```tsx
<Dropdown
  items={[
    { children: "Item 1", href: "/item-1" },
    { children: "Item 2", href: "/item-2" },
    { children: "Item 3", href: "/item-3" },
  ]}
  // replace default anchor with a custom link component using the renderLink prop
  // renderLink={(item) => <Link href={item.href}>{item.children}</Link>}
>
  Actions
</Dropdown>
```

### With Sections Dropdown

Use when dropdown items are logically grouped into categories that benefit from visual separation. Use submenus when certain actions should be discoverable without cluttering the main list.

```tsx
<Dropdown
  items={[
    {
      slotHeading: "Navigation",
      items: [
        { children: "Dashboard", onSelect: handleSelect },
        { children: "Settings", onSelect: handleSelect },
      ],
    },
    {
      slotHeading: "Account",
      items: [
        { children: "Profile", onSelect: handleSelect },
        { children: "Delete", danger: true, onSelect: handleSelect },
      ],
    },
    {
      kind: "sub",
      children: "Support",
      items: [
        { children: "FAQ", href: "/faq" },
        { children: "Contact Us", href: "/contact-us" },
      ],
    },
  ]}
>
  Menu
</Dropdown>
```

### Filterable Dropdown

Use when the dropdown contains many items and the user benefits from searching to find their selection quickly.

```tsx
<Dropdown filterable items={[{ children: "United States" }, { children: "Canada" }, { children: "United Kingdom" }, { children: "Germany" }, { children: "Japan" }]} onItemSelect={handleItemSelect}>
  Country
</Dropdown>
```

### Icon Only Trigger Dropdown

Use an icon child with showChevron={false} for kebab/overflow menus where space is constrained. Always supply an aria-label.

```tsx
<Dropdown
  showChevron={false}
  aria-label="More actions"
  items={[
    { children: "Edit", onSelect: handleSelect },
    { children: "Duplicate", onSelect: handleSelect },
    { children: "Delete", danger: true, onSelect: handleSelect },
  ]}
>
  <MenuIcon />
</Dropdown>
```

### Server Actions Dropdown

Wrap composed primitives in a <form> and pass formAction/formMethod on each DropdownItem so the menu submits to different endpoints without JavaScript.

```tsx
<form>
  <DropdownRoot tabBehavior="close">
    <DropdownTrigger>Task Actions</DropdownTrigger>
    <DropdownContent>
      <DropdownItem formAction="/api/tasks/complete" formMethod="post">
        Mark as Complete
      </DropdownItem>
      <DropdownItem formAction="/api/tasks/reassign" formMethod="post">
        Reassign
      </DropdownItem>
      <DropdownItem danger formAction="/api/tasks/delete" formMethod="post">
        Delete Task
      </DropdownItem>
    </DropdownContent>
  </DropdownRoot>
</form>
```

### Composed

```tsx
<DropdownRoot tabBehavior="close">
  <DropdownTrigger>Actions</DropdownTrigger>
  {/* or for kebab menu: <DropdownTrigger showChevron={false}><MoreVert /></DropdownTrigger> */}
  <DropdownContent>
    <DropdownItem
      slotStart={<Pencil />}
      // onSelect={handleSelect}
    >
      Rename
    </DropdownItem>
    <DropdownItem
      slotStart={<Trash />}
      danger
      // onSelect={handleSelect}
    >
      Delete
    </DropdownItem>
  </DropdownContent>
</DropdownRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `DropdownEntry[]` | - | Entries rendered inside the dropdown. |
| align | `"center" \| "start" \| "end"` | `"start"` | Alignment of the content relative to the trigger. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ DropdownContent?: NativeElementAttributes<"menu", typeof DropdownContent>; MenuSearch?: NativeElementAttributes<"input", typeof MenuSearch> }` | - | Native HTML attributes forwarded to the internal composed components. |
| defaultFilterValue | `string` | - | Initial value of the dropdown search input when filterable. |
| defaultOpen | `boolean` | `false` | Initial open state when uncontrolled. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces. |
| disabled | `boolean` | - | Disables interaction with the dropdown. |
| filterable | `boolean` | `false` | Renders a search input that filters the items. |
| filterMatchFn | `"disable" \| ((matchTerm: string, value: string) => boolean)` | - | Overrides the default substring matching used to filter items. |
| filterValue | `string` | - | Controlled value of the search input. Pair with `onFilterChange`. |
| id | `string` | - | Stable identifier used to derive the popover id and CSS anchor name. Provide one to keep generated ids consistent across server and client renders. |
| modal | `boolean` | `false` | Renders the menu as a modal that blocks interaction with the rest of the page. |
| onCloseAutoFocus | `(event: Event) => void` | - | Called when focus moves back to the trigger after closing. Call `event.preventDefault()` to suppress the focus shift. |
| onEscapeKeyDown | `(event: KeyboardEvent) => void` | - | Called when the Escape key is pressed. Call `event.preventDefault()` to prevent the default close behavior. |
| onFilterChange | `(value: string) => void` | - | Called when the search value changes. |
| onInteractOutside | `(event: Event) => void` | - | Called when any interaction occurs outside the content bounds. Call `event.preventDefault()` to keep the dropdown open. |
| onItemCheckedChange | `(item: Omit<MenuCheckboxItemEntry, "attributes">, checked: "indeterminate" \| boolean) => void` | - | Called when a checkbox item's checked state changes. |
| onItemSelect | `(event: Event, item: { href?: string } & Omit<MenuDefaultItemEntry, "attributes"> \| Omit<MenuCheckboxItemEntry, "attributes"> \| Omit<MenuRadioItemEntry, "attributes">) => void` | - | Called when an item is selected via mouse or keyboard. |
| onOpenChange | `(open: boolean) => void` | - | Called when the open state changes. |
| onPointerDownOutside | `(event: Event) => void` | - | Called when a pointer event occurs outside the content bounds. Call `event.preventDefault()` to keep the dropdown open. |
| open | `boolean` | - | Controlled open state. Pair with `onOpenChange`. |
| portal | `boolean` | `false` | Renders the floating content in a React portal. |
| portalContainer | `HTMLElement` | `document.body` | Container for portaled floating content when no nearer portal target exists. |
| ref | `Ref<HTMLButtonElement>` | - |  |
| renderLink | `(item: Omit<DropdownDefaultItemEntry, "href"> & { href: string; }) => ReactNode` | - | Custom renderer for items with an `href`. Use to swap in a framework-specific link component such as Next.js `<Link>`. |
| showChevron | `boolean` | `true` | Renders a chevron that reflects the current open state. Set to `false` for icon-only kebab/overflow triggers — the icon already signals a menu. Keep `true` for standalone text triggers. |
| side | `"right" \| "bottom" \| "left" \| "top"` | `"bottom"` | Preferred side of the trigger to render against. Flips automatically when there is not enough space. |
| size | `"small" \| "tiny" \| "medium" \| "large"` | `"medium"` | Size applied to the dropdown and its trigger. |
| ~~tabBehavior~~ | `"close" \| "navigate"` | `"close"` | Whether Tab closes the popup or moves through its enabled items while keeping it open. *(Deprecated: - will be removed in the next major version and only allow "close")* |

`* = required prop`

## Exports

- Components: `Dropdown`, `DropdownCheckboxItem`, `DropdownContent`, `DropdownHeading`, `DropdownItem`, `DropdownRadioGroup`, `DropdownRadioGroupItem`, `DropdownRoot`, `DropdownSection`, `DropdownSub`, `DropdownSubContent`, `DropdownSubTrigger`, `DropdownTrigger`
- Hooks: `useDelayedToggle`, `useDropdownContext`, `useDropdownSubContext`
- Constants: `SUBMENU_CLOSE_DELAY_MS`, `SUBMENU_OPEN_DELAY_MS`
- Types: `BaseDropdownItem`, `DropdownCheckboxItemEntry`, `DropdownCheckboxItemProps`, `DropdownContentProps`, `DropdownContextType`, `DropdownDefaultItemEntry`, `DropdownDividerItemEntry`, `DropdownEntry`, `DropdownHeadingProps`, `DropdownItemProps`, `DropdownKind`, `DropdownProps`, `DropdownRadioGroupEntry`, `DropdownRadioGroupItemProps`, `DropdownRadioGroupProps`, `DropdownRadioItemEntry`, `DropdownRenderLinkItem`, `DropdownRootProps`, `DropdownSectionEntry`, `DropdownSectionProps`, `DropdownSubContentProps`, `DropdownSubContextType`, `DropdownSubProps`, `DropdownSubSection`, `DropdownSubTriggerProps`, `DropdownTriggerProps`

## See also

- [Menu](./Menu.md)
