# VerticalNav

A vertical navigation menu of single links and nested link groups, typically rendered in a sidebar. Use as section-level navigation paired with a HorizontalNav for app-wide chrome, or as the sole navigation for apps with deep hierarchies and few global destinations.

## Notes

- Choose the right nav primitive: VerticalNav-as-sidebar (paired with HorizontalNav for app-wide chrome) suits apps with many sections or deep IA; VerticalNav-alone fits deep hierarchies with few global destinations; HorizontalNav-alone fits apps with seven or fewer top-level sections.

- Shell/layout integration for sidebar layouts: AppBar must be sticky top, the aside needs `overflow-x-hidden`, the sticky aside's `top-*` should match the AppBar height, and avoid wrapping the nav in extra `px-*` containers (it breaks mobile drawer behavior).

- VerticalNav supports a maximum of two levels (parent -> child). Do not add a third nesting level; flatten the structure or use breadcrumbs for deeper hierarchies.

- Omit `slotIcon` by default — labels alone are sufficient for most internal apps. Icons act as scanning anchors only when there are many parallel sections with weak label differentiation. If two nav items would use similar-looking icons, drop the icons; they hurt scanning.

- Never render nav items as disabled or aria-disabled for role-gated destinations. Remove inaccessible items from the items array entirely.

- Keep labels short (one to three words) and noun-based ("Clusters", "Access Control"); avoid verbs like "Manage Clusters".

- Abbreviate labels only when the abbreviation is universally understood ("GPU" is fine, "Cfg" is not).

- If any top-level item sets slotIcon, set it on every top-level sibling so the left edge aligns; sub-items typically omit icons.

- Nest sub-items only when they are frequently accessed and users switch between them often. Route to breadcrumbs or tabs when sub-items are only ever reached from the parent. If the IA needs three or more levels, reconsider the IA rather than adding nav depth.

- Anti-patterns to avoid: using VerticalNav for simple apps with few destinations, nesting past two levels, omitting `renderLink` on framework-router apps (causes full page reloads), and wrapping the nav in extra horizontal padding containers.

- Pair with `renderLink` to integrate a framework router so navigation stays client-side.

- When a `VerticalNavItem` triggers an in-page action (selecting a chat, opening a panel) rather than navigating to a URL, render it with `asChild` onto a `<button type="button">`.

- `VerticalNav` (and `VerticalNavRoot` when composing) provides its own chrome - bg-surface-base, border-r, h-full, 240px wide (56px when `collapsed`). Set `--nv-vertical-nav-width` on an ancestor to change the expanded width, e.g. `260px` for the new design. NEVER reapply these styles on a wrapping `<aside>` element.

- Group related top-level items under a heading with an entry of `{ id, slotHeading, items }` (composed: `VerticalNavGroup` > `VerticalNavHeading` + `VerticalNavGroupList`). The heading is a non-interactive label; set `collapsible: true` on the entry (or `VerticalNavGroup`) when it should collapse its items. The items stay top-level rows, not indented. Never make a heading a link. Use an item with `subItems` instead when the parent is itself a destination-like row with an icon and nested children.

- Separate top-level rows or groups with a rule using a `{ kind: "divider" }` entry (composed: `VerticalNavDivider` in the `VerticalNavList`). It brings its own 16px of space above and below, so use it in place of a heading's spacing, not alongside extra margins.

- Set `collapsed` to render a 56px icon rail: only top-level rows show, as icons, so give every top-level item a `slotIcon`. Labels remain the accessible names, headings become rules, dividers stay, collapsible headed groups always show their items, and items with `subItems` open as flyouts on hover, click, or Enter. Flyouts are fixed-position, so keep the nav out of any ancestor that sets `transform`, `filter`, or `contain` and also clips its overflow.

- For pages that need maximum horizontal space (wide tables, multi-column dashboards), let an `AppBarExpanderButton` toggle between the full nav and the `collapsed` rail, or hide the nav entirely. On narrow viewports, render the nav inside a `SidePanel` opened from the same button instead, so it overlays the content with a blanket and traps focus.

## Examples

### Basic Vertical Nav

Use for sidebar navigation with a flat list of destinations.

```tsx
<VerticalNav
  items={[
    { id: "home", children: "Home", href: "/" },
    {
      id: "settings",
      children: "Settings",
      subItems: [
        {
          id: "profile",
          active: true,
          children: "Profile",
          href: "/settings/profile",
        },
        {
          id: "account",
          children: "Account",
          href: "/settings/account",
        },
      ],
    },
  ]}
/>
```

### Default Closed Vertical Nav

Set defaultOpen: false on a group to start collapsed — use when the nav has many sections and you want to reduce visual noise on first load.

```tsx
<VerticalNav
  items={[
    {
      id: "datacenter",
      children: "Data Center",
      defaultOpen: false,
      slotIcon: <Icon name="document" />,
      subItems: [
        {
          id: "datacenter-home",
          children: "Home",
          href: "/data-center",
        },
        {
          id: "datacenter-products",
          children: "Products",
          href: "/data-center/products",
        },
      ],
    },
    {
      id: "workspace",
      children: "Workspace",
      defaultOpen: false,
      slotIcon: <Icon name="folder-closed" />,
      subItems: [
        {
          id: "workspace-apps",
          children: "Apps",
          href: "/workspace/apps",
        },
      ],
    },
  ]}
/>
```

### Heading Groups Vertical Nav

Group related destinations under a heading with `{ id, slotHeading, items }`; the heading labels the group's list. Set `collapsible: true` when the heading should collapse its group, and add a `{ kind: "divider" }` entry to separate rows with a rule instead of a heading.

```tsx
<VerticalNav
  items={[
    { id: "dashboard", children: "Dashboard", href: "/", active: true },
    {
      id: "models",
      slotHeading: "Models",
      items: [
        {
          id: "custom-models",
          children: "Custom Models",
          href: "/models/custom",
        },
        { id: "base-models", children: "Base Models", href: "/models/base" },
      ],
    },
    {
      id: "data",
      slotHeading: "Data",
      collapsible: true,
      items: [
        { id: "datasets", children: "Datasets", href: "/data/datasets" },
        {
          id: "safe-synthesizer",
          children: "Safe Synthesizer",
          href: "/data/safe-synthesizer",
        },
      ],
    },
    { kind: "divider" },
    { id: "evaluations", children: "Evaluations", href: "/evaluations" },
  ]}
/>
```

### Icon Rail Vertical Nav

Set `collapsed` for a 56px icon rail when the page needs the width. Give every top-level item a `slotIcon`: the labels stay as accessible names, and collapsible groups open as flyouts on hover, click, or Enter.

```tsx
<VerticalNav
  collapsed
  aria-label="Main"
  items={[
    {
      id: "home",
      children: "Home",
      href: "/",
      active: true,
      slotIcon: <Icon name="document" />,
    },
    {
      id: "tasks",
      children: "My tasks",
      href: "/tasks",
      slotIcon: <Icon name="check-circle" />,
    },
    {
      id: "insights",
      children: "Insights",
      slotIcon: <Icon name="folder-closed" />,
      subItems: [
        {
          id: "reporting",
          children: "Reporting",
          href: "/insights/reporting",
        },
        { id: "goals", children: "Goals", href: "/insights/goals" },
      ],
    },
  ]}
/>
```

### Render Link Vertical Nav

Use renderLink to integrate a framework router (Next.js Link, React Router NavLink, etc.) — a single renderer is applied to both primary and secondary items via the asChild pattern, so styling stays consistent.

```tsx
<VerticalNav
  items={[
    { id: "home", children: "Home", href: "/" },
    {
      id: "settings",
      children: "Settings",
      subItems: [
        {
          id: "profile",
          children: "Profile",
          href: "/settings/profile",
        },
        {
          id: "account",
          children: "Account",
          href: "/settings/account",
        },
      ],
    },
  ]}
  renderLink={(item) => (
    <a href={item.href} target="_blank" rel="noopener noreferrer">
      {item.children}
    </a>
  )}
/>
```

### Custom Footer Vertical Nav

Drop down to the composed primitives when the nav needs extra chrome (header, footer, version badge). Padding lives on the list, so a bordered footer extends edge-to-edge across the full nav width.

```tsx
<VerticalNavRoot className="flex flex-col">
  <VerticalNavList className="min-h-0 flex-1">
    <VerticalNavListItem>
      <VerticalNavItem href="/data-center">Data Center</VerticalNavItem>
    </VerticalNavListItem>
    <VerticalNavListItem>
      <VerticalNavItem href="/workspace" active>
        Workspace
      </VerticalNavItem>
    </VerticalNavListItem>
    <VerticalNavListItem>
      <VerticalNavItem href="/settings">Settings</VerticalNavItem>
    </VerticalNavListItem>
  </VerticalNavList>
  <div className="shrink-0 border-t-1 border-t-base px-4 py-3 text-center text-label-regular-sm text-secondary">Version 1.2.3</div>
</VerticalNavRoot>
```

### Composed

```tsx
<VerticalNavRoot>
  <VerticalNavList>
    <VerticalNavListItem>
      <VerticalNavItem href="/" active>
        Home
      </VerticalNavItem>
    </VerticalNavListItem>
    <VerticalNavListItem>
      {/* action VerticalNavItem should be asChild onto a <button> */}
      <VerticalNavItem asChild>
        <button type="button">See License Agreement</button>
      </VerticalNavItem>
    </VerticalNavListItem>
    <VerticalNavListItem>
      <VerticalNavCollapsibleSection defaultOpen>
        <VerticalNavItem asChild slotEnd={<Icon name="chevron-down" />}>
          <VerticalNavCollapsibleTrigger>Settings</VerticalNavCollapsibleTrigger>
        </VerticalNavItem>
        <VerticalNavCollapsibleContent>
          {/* titles the section's flyout in the icon rail; hidden otherwise */}
          <VerticalNavHeading>Settings</VerticalNavHeading>
          <VerticalNavSubList>
            <VerticalNavSubListItem>
              <VerticalNavItem kind="secondary" href="/settings/profile">
                Profile
              </VerticalNavItem>
            </VerticalNavSubListItem>
            <VerticalNavSubListItem>
              <VerticalNavItem kind="secondary" href="/settings/account">
                Account
              </VerticalNavItem>
            </VerticalNavSubListItem>
          </VerticalNavSubList>
        </VerticalNavCollapsibleContent>
      </VerticalNavCollapsibleSection>
    </VerticalNavListItem>
    <VerticalNavDivider />
    <VerticalNavGroup collapsible defaultOpen>
      <VerticalNavHeading>Data</VerticalNavHeading>
      <VerticalNavGroupList>
        <VerticalNavListItem>
          <VerticalNavItem href="/datasets">Datasets</VerticalNavItem>
        </VerticalNavListItem>
      </VerticalNavGroupList>
    </VerticalNavGroup>
  </VerticalNavList>
</VerticalNavRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `VerticalNavBaseEntry[]` | - | Navigation items, and headed groups of items, rendered in order. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `VerticalNavAttributes` | - | Attributes to apply to the internal VerticalNav composed components. |
| collapsed | `boolean` | `false` | Renders the nav as a 56px icon rail. Top-level rows show only their icon and keep their label as their accessible name, headings become dividers, and each collapsible section opens as a flyout beside the rail on hover, click, or Enter. |
| renderLink | `(item: Omit<VerticalNavBaseLeafItem, "href"> & { href: string; } \| Omit<VerticalNavBaseSubItem, "href"> & { href: string; }) => ReactNode` | - | Custom renderer for navigation links. Used for both primary leaf items and secondary sub-items — group items always render as a collapsible trigger and never call `renderLink`. The result is wrapped in `VerticalNavItem` with `asChild`, so the consumer's element gets all styling/behavior. |

`* = required prop`

## Exports

- Components: `VerticalNav`, `VerticalNavCollapsibleContent`, `VerticalNavCollapsibleSection`, `VerticalNavCollapsibleTrigger`, `VerticalNavDivider`, `VerticalNavGroup`, `VerticalNavGroupList`, `VerticalNavHeading`, `VerticalNavItem`, `VerticalNavList`, `VerticalNavListItem`, `VerticalNavRoot`, `VerticalNavSubList`, `VerticalNavSubListItem`
- Types: `VerticalNavAttributes`, `VerticalNavBaseCollapsibleHeadingGroup`, `VerticalNavBaseDivider`, `VerticalNavBaseEntry`, `VerticalNavBaseGroupItem`, `VerticalNavBaseHeadingGroup`, `VerticalNavBaseItem`, `VerticalNavBaseLeafItem`, `VerticalNavBaseStaticHeadingGroup`, `VerticalNavBaseSubItem`, `VerticalNavCollapsibleContentProps`, `VerticalNavCollapsibleSectionProps`, `VerticalNavCollapsibleTriggerProps`, `VerticalNavDividerProps`, `VerticalNavGroupListProps`, `VerticalNavGroupProps`, `VerticalNavHeadingGroupAttributes`, `VerticalNavHeadingProps`, `VerticalNavItemAttributes`, `VerticalNavItemProps`, `VerticalNavListItemProps`, `VerticalNavListProps`, `VerticalNavProps`, `VerticalNavRenderLinkItem`, `VerticalNavRootProps`, `VerticalNavSubItemAttributes`, `VerticalNavSubListItemProps`, `VerticalNavSubListProps`

## See also

- [HorizontalNav](./HorizontalNav.md)
- [AppBar](./AppBar.md)
