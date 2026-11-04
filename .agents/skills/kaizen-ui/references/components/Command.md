# Command

A modal menu that lets users search for and run actions or jump to pages from the keyboard. Opens over the page, filters its items as the user types, and closes when an item is chosen.

## Notes

- Use Command for an app-wide palette of actions and destinations, usually opened with a global shortcut such as Meta+K. For a menu attached to a button, use Dropdown. For choosing a form value, use Combobox or Select.

- Command always renders in a modal dialog. To open it from a global shortcut, control it with `open` and `onOpenChange` and listen for the key yourself. Put `aria-keyshortcuts` on any visible trigger.

- Group items into sections under short `slotHeading` labels such as "Suggestions", "Navigation", or "Settings". Separate sections with `{ kind: "divider" }`.

- Show an action's shortcut in `slotEnd` as `<Kbd kind="solid" size="small" items={[...]} />`, and name it with `aria-keyshortcuts` in the item's `attributes`. Show it only if the shortcut also works outside the menu.

- Use `href` for items that navigate, and pass `renderLink` so framework routing stays client-side. Use `onSelect` for items that perform an action.

- Set `filterValue` when users might search with words that are not in the label, such as "preferences" for a "Settings" item.

- Within a section, either every item has a `slotStart` icon or none does.

## Examples

### Basic Command

Use for an app-wide palette of actions and destinations. Give actions an `onSelect` and destinations an `href`, and group them under short headings.

```tsx
<Command
  slotTrigger={<Button kind="secondary">Open command menu</Button>}
  items={[
    {
      slotHeading: "Suggestions",
      items: [
        {
          children: "New document",
          // onSelect: createDocument,
          slotStart: <Icon name="document" />,
          slotEnd: <Kbd items={["⌘", "N"]} kind="solid" size="small" />,
        },
        {
          children: "Upload file",
          // onSelect: uploadFile,
          slotStart: <Icon name="paperclip" />,
          slotEnd: <Kbd items={["⌘", "U"]} kind="solid" size="small" />,
        },
        {
          children: "Refresh data",
          // onSelect: refreshData,
          slotStart: <Icon name="refresh" />,
        },
      ],
    },
    { kind: "divider" },
    {
      slotHeading: "Navigation",
      items: [
        {
          children: "Calendar",
          href: "/calendar",
          slotStart: <Icon name="calendar" />,
        },
        {
          children: "Notifications",
          href: "/notifications",
          slotStart: <Icon name="bell" />,
        },
        {
          children: "Projects",
          href: "/projects",
          slotStart: <Icon name="folder-closed" />,
        },
      ],
    },
  ]}
/>
```

### Keyboard Shortcut Command

Control `open` to toggle the menu from a global shortcut. Listen for the key on the document, and name the shortcut on the visible trigger with `aria-keyshortcuts`.

```tsx
() => {
  const [open, setOpen] = useState(false);

  /** @remarks the shortcut works anywhere on the page, so it listens on the document rather than on an element */
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "k" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        setOpen((current) => !current);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <Command
      open={open}
      onOpenChange={setOpen}
      slotTrigger={
        <Button aria-keyshortcuts="Meta+K Control+K" kind="secondary">
          Search
          <Kbd items={["⌘", "K"]} kind="flat" size="small" />
        </Button>
      }
      items={[
        { children: "Calendar", href: "/calendar" },
        { children: "Notifications", href: "/notifications" },
        {
          children: "Settings",
          filterValue: "settings preferences",
          href: "/settings",
        },
      ]}
    />
  );
};
```

### Composed

Use the composed parts when you need to control each region yourself, such as rendering an item as your router's link through `asChild`.

```tsx
<CommandRoot>
  <CommandTrigger asChild>
    <Button kind="secondary">Search docs</Button>
  </CommandTrigger>
  <CommandDialog aria-label="Search docs">
    <CommandContent>
      <CommandInput placeholder="Search pages..." />
      <CommandList emptyStateMessage="No pages found">
        <CommandSection slotHeading="Guides">
          <CommandItem asChild slotDescription="Install the packages and render a first component">
            <a href="/guides/getting-started">Getting started</a>
          </CommandItem>
          <CommandItem asChild slotDescription="Switch between light and dark themes">
            <a href="/guides/theming">Theming</a>
          </CommandItem>
        </CommandSection>
        <CommandDivider />
        <CommandSection slotHeading="Components">
          <CommandItem asChild filterValue="Modal dialog overlay">
            <a href="/components/modal">Modal</a>
          </CommandItem>
          <CommandItem asChild filterValue="Side panel drawer sheet">
            <a href="/components/side-panel">Side Panel</a>
          </CommandItem>
        </CommandSection>
      </CommandList>
    </CommandContent>
  </CommandDialog>
</CommandRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `(CommandItemEntry \| { kind?: "section"; slotHeading?: ReactNode; items: CommandItemEntry[] } \| { kind: "divider" })[]` | - | Items, sections, and dividers rendered in the list, in order. |
| aria-label | `string` | `"Command menu"` | Accessible name of the dialog. Pass `aria-labelledby` instead when a visible heading names it. |
| attributes | `{ CommandTrigger?: NativeElementAttributes<"button", typeof CommandTrigger>; CommandAnchorTrigger?: NativeElementAttributes<"a", typeof CommandAnchorTrigger>; CommandDialog?: NativeElementAttributes<"dialog", typeof CommandDialog>; CommandInput?: Omit<CommandInputProps, "placeholder">; CommandList?: NativeElementAttributes<"div", typeof CommandList> }` | - | Native HTML attributes forwarded to the internal composed components. |
| closeOnClickOutside | `boolean` | `true` | Whether clicking the backdrop closes the menu. |
| defaultFilterValue | `string` | - | Initial search value when the menu is uncontrolled. The search resets each time the menu opens. |
| defaultOpen | `boolean` | `false` | Initial open state when the dialog is uncontrolled. |
| emptyStateMessage | `string` | `"No results found"` | Message shown in place of the items when the search matches nothing, and announced to screen readers. |
| filterMatchFn | `"disable" \| ((matchTerm: string, value: string) => boolean)` | - | Custom matcher used to decide whether an item is visible for the current search value. Defaults to a case- and diacritic-insensitive substring match. Set to `"disable"` to show every item, for example when results come from a server. |
| filterValue | `string` | - | Controlled search value. Pair with `onFilterChange`, and clear it when the menu closes, since only an uncontrolled search resets on its own. |
| forceMount | `boolean` | - | Ensures dialog content remains mounted when the dialog is not open. |
| id | `string` | - | HTML `id` attribute applied to the `<dialog>` element. |
| onEscapeKeyDown | `(event: KeyboardEvent) => void` | - | Called when Escape is pressed inside the dialog. Call `event.preventDefault()` to keep the dialog open. |
| onFilterChange | `(value: string) => void` | - | Called with the new search value as the user types. |
| onInteractOutside | `(event: MouseEvent) => void` | - | Called when any interaction occurs outside the dialog. Call `event.preventDefault()` to keep the menu open. |
| onItemSelect | `(event: Event, item: CommandItemEntry) => void` | - | Called when any item is chosen, receiving the underlying event and the entry. Call `event.preventDefault()` to keep the menu open. |
| onOpenChange | `(open: boolean, request: { cancel: () => void; }) => void` | - | Callback invoked whenever the dialog requests an open state change, in both controlled and uncontrolled modes. A controlled dialog requests `false` before its exit animation, so keeping `open` as `true` keeps it open. When refusing, also call `request.cancel()` so a later close does not reuse the refused return value. Controlled prop changes made by the consumer are not echoed back through this callback. |
| onPointerDownOutside | `(event: MouseEvent) => void` | - | Called when a pointer event occurs outside the dialog. Call `event.preventDefault()` to keep the menu open. |
| open | `boolean` | - | Controlled open state. When provided, the hook operates in controlled mode and `defaultOpen` is ignored. The consumer is responsible for updating this value in response to {@link UseDialogOptions.onOpenChange} callbacks, and the dialog stays open until it becomes `false`. |
| placeholder | `string` | `"Type a command or search..."` | Hint shown while the field is empty. Also the field's accessible name unless `aria-label` is set. |
| portal | `boolean` | `false` | Renders the floating content in a React portal. |
| portalContainer | `HTMLElement` | `document.body` | Container for portaled floating content when no nearer portal target exists. |
| renderLink | `(item: Omit<CommandItemEntry, "href"> & { href: string; }) => ReactNode` | - | Custom renderer for items with an `href`. Use to swap in a framework-specific link component such as Next.js `<Link>`. |
| slotAnchorTrigger | `ReactNode` | - | Anchor-based trigger that opens the menu on click and falls back to navigation when JavaScript is unavailable. Requires `id`. |
| slotTrigger | `ReactNode` | - | Trigger that opens the menu when activated. Omit it to open the menu from your own state with `open` and `onOpenChange`. |

`* = required prop`

## Exports

- Components: `Command`, `CommandAnchorTrigger`, `CommandContent`, `CommandDialog`, `CommandDivider`, `CommandHeading`, `CommandInput`, `CommandItem`, `CommandList`, `CommandRoot`, `CommandSection`, `CommandTrigger`
- Types: `CommandAnchorTriggerProps`, `CommandContentProps`, `CommandDialogProps`, `CommandDividerEntry`, `CommandDividerProps`, `CommandEntry`, `CommandHeadingProps`, `CommandInputProps`, `CommandItemEntry`, `CommandItemProps`, `CommandListProps`, `CommandProps`, `CommandRenderLinkItem`, `CommandRootProps`, `CommandSectionEntry`, `CommandSectionProps`, `CommandTriggerProps`

## See also

- [Dropdown](./Dropdown.md)
- [Combobox](./Combobox.md)
- [Kbd](./Kbd.md)
