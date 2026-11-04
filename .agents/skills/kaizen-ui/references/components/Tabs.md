# Tabs

Organizes related content into sections that share the same context and switches between them in place. Use for managing settings, comparing data, or any group of mutually exclusive views the user moves between without leaving the page.

## Notes

- Use `activationMode="automatic"` only when switching panels is lightweight enough to happen immediately as focus moves; keep the default "manual" for anything that fetches data or causes layout shift.

- Reach for Tabs to switch between different content panels on the same page. Use SegmentedControl for toggling between views of the same data (list vs grid, chart vs table), and HorizontalNav/VerticalNav for top-level routing.

- When tabs map to URL routes, pass `href` on each item, along with `renderLink` to swap the underlying anchor for a framework router component (Next.js `Link`, React Router `Link`, etc.).

- Do not nest `kind="primary"` tabs inside another primary tab group. Use `kind="secondary"` or `kind="tertiary"` for nested tab groups so the hierarchy reads.

- For status filter tabs (e.g. "All", "Draft", "Completed"), include an "All" option first and place count badges in the label so users can assess distribution.

- If any item sets a leading icon, set it on every item in the same tabs list so the visual rhythm stays even.

- Tabs serve two patterns: distinct content sections (e.g. "Overview", "Configuration", "Logs") where the whole panel changes, and filtered views of the same dataset (e.g. "All", "Draft", "Completed") where the row structure is fixed and only the contents change.

- Status filter tabs are for a single predefined attribute. If users need to combine multiple filter criteria, use a dedicated filter toolbar or Dropdown filter controls instead of tabs.

- Icons on tabs are optional and must reinforce — not replace — the label. Always include text (icon-only tabs are ambiguous), and place the icon leading the label.

- Managed content (`slotContent` per item) is the default. Switch to controlled tabs when content must survive tab switches (form state, scroll position) or when content is loaded asynchronously.

- Avoid excessive tabs: scrollable overflow exists but users may not scroll to find hidden tabs. Reconsider hierarchy or switch to VerticalNav when the list is long.

- Prefer HorizontalNav/VerticalNav over Tabs for top-level navigation

## Examples

### Basic Tabs

Use when content is divided into related sections that users switch between.

```tsx
<Tabs
  items={[
    {
      children: "Account",
      slotContent: "Account settings content",
      value: "account",
    },
    {
      children: "Password",
      slotContent: "Password settings content",
      value: "password",
    },
    {
      children: "Notifications",
      slotContent: "Notification preferences",
      value: "notifications",
    },
    {
      children: "Disabled",
      slotContent: "Disabled tab content",
      value: "disabled",
      disabled: true,
    },
    {
      children: "Another",
      slotContent: "Another tab content",
      value: "another",
    },
  ]}
/>
```

### Status Filter Tabs

Use Tabs to filter a single dataset by status. Lead with an "All" tab so users can clear the filter, and pair labels with count Badges so distribution is visible before clicking.

```tsx
<Tabs
  defaultValue="all"
  items={[
    {
      value: "all",
      children: (
        <Flex align="center" gap="2">
          All <Badge>128</Badge>
        </Flex>
      ),
    },
    {
      value: "draft",
      children: (
        <Flex align="center" gap="2">
          Draft <Badge>12</Badge>
        </Flex>
      ),
    },
    {
      value: "in-progress",
      children: (
        <Flex align="center" gap="2">
          In Progress <Badge color="yellow">8</Badge>
        </Flex>
      ),
    },
    {
      value: "completed",
      children: (
        <Flex align="center" gap="2">
          Completed <Badge color="green">108</Badge>
        </Flex>
      ),
    },
  ]}
/>
```

### With Slots Tabs

Use `slotStart` / `slotEnd` to render presentational labels or badges flanking the triggers inside the tabs list.

```tsx
<Tabs
  items={[
    { children: "Tab 1", slotContent: "Content 1", value: "1" },
    { children: "Tab 2", slotContent: "Content 2", value: "2" },
  ]}
  slotStart={
    <span role="presentation" className="px-2 font-bold">
      Section
    </span>
  }
  slotEnd={
    <span role="presentation" className="ml-auto">
      2 tabs
    </span>
  }
/>
```

### Controlled Tabs

Use controlled mode when tab content lives outside the `Tabs` component — e.g. when content is large or shared with other UI. Pass `value` + `onValueChange` and render the panels yourself instead of using `slotContent` per item.

```tsx
() => {
  const [value, setValue] = useState("account");
  return (
    <Flex direction="col" gap="3">
      <Tabs
        items={[
          { children: "Account", value: "account" },
          { children: "Notifications", value: "notifications" },
          { children: "Billing", value: "billing" },
        ]}
        value={value}
        onValueChange={setValue}
      />
      {value === "account" && <Text>Account settings content</Text>}
      {value === "notifications" && <Text>Notification preferences content</Text>}
      {value === "billing" && <Text>Billing details content</Text>}
    </Flex>
  );
};
```

### Composed

```tsx
<TabsRoot defaultValue="tab1">
  <TabsList>
    <TabsTrigger value="tab1">Tab 1</TabsTrigger>
    <TabsTrigger value="tab2">Tab 2</TabsTrigger>
  </TabsList>
  <TabsContent value="tab1">Content for tab 1</TabsContent>
  <TabsContent value="tab2">Content for tab 2</TabsContent>
</TabsRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `{ asChild?: boolean; children: ReactNode; slotContent?: ReactNode; value: string; disabled?: boolean; href?: string }[]` | - | Tab definitions rendered in order. |
| activationMode | `"manual" \| "automatic"` | `"manual"` | Whether tabs activate as keyboard focus moves to them, or only when Enter or Space is pressed. - "manual" - Arrow keys move focus; Enter or Space activates the focused tab. Use when switching panels is expensive. - "automatic" - Arrow keys both move focus and activate the focused tab. Use when switching is instant. |
| attributes | `{ TabsList?: NativeElementAttributes<"div", typeof TabsList> }` | - | Native HTML attributes forwarded to the internal composed components. |
| contentPadding | `"none" \| "default"` | `"default"` | Padding applied to generated tab panels. |
| defaultValue | `string` | `items[0]?.value` | Initial active tab value when uncontrolled. |
| hideOverflowButtons | `boolean` | `false` | Hides the chevron buttons rendered when the list overflows horizontally. |
| kind | `"primary" \| "secondary" \| "tertiary"` | `"primary"` | The visual hierarchy of the tabs. - "primary" - Default style with bottom-border indicators, used for the top-level sections of a view. - "secondary" - Pill-shaped tabs with a filled background, used to divide content within a primary section. - "tertiary" - Minimal, text-only style for optional or supplementary content. |
| onValueChange | `(value: string) => void` | - | Called when the active tab changes. |
| ref | `Ref<HTMLDivElement>` | - |  |
| renderLink | `(item: Omit<TabItem, "href"> & { href: string; }) => ReactNode` | - | Renders a custom link element for each tab with an `href`. When items have `href`, the root element becomes a `<nav>` landmark. |
| slotEnd | `ReactNode` | - | Content rendered after the tab triggers inside the list. |
| slotStart | `ReactNode` | - | Content rendered before the tab triggers inside the list. |
| value | `string` | - | Controlled active tab value. Pair with `onValueChange`. |
| visibleRange | `number[]` | - | Restricts the visible triggers to the given child indices, inserting an ellipsis for each gap. For example, `[1,2,3,8,9,10]` shows items 1-3, an ellipsis, then 8-10. |

`* = required prop`

## Exports

- Components: `Tabs`, `TabsContent`, `TabsList`, `TabsRoot`, `TabsTrigger`
- Hooks: `useTabsContext`
- Types: `TabItem`, `TabRenderLinkItem`, `TabsContentProps`, `TabsListProps`, `TabsProps`, `TabsPropsBase`, `TabsPropsDivRoot`, `TabsPropsNavRoot`, `TabsRootProps`, `TabsTriggerProps`

## See also

- [SegmentedControl](./SegmentedControl.md)
- [HorizontalNav](./HorizontalNav.md)
- [VerticalNav](./VerticalNav.md)
