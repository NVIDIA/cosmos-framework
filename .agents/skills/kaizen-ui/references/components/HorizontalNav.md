# HorizontalNav

A horizontal navigation bar that organizes links to top-level sections of an application or site.

## Notes

- Use `disabled` only for transient unavailability (a destination temporarily offline, under maintenance, or pending setup). For role-gated destinations the current user cannot access, remove the item from the items array entirely instead.

- Place HorizontalNav inside AppBar's `children` as the primary top-level nav, when using it for app-wide navigation.

- Choose the nav shape by section count: ≤7 sections → HorizontalNav alone; 7+ sections or section-level sub-nav → HorizontalNav paired with VerticalNav; deep hierarchy with few global destinations → VerticalNav alone.

- The real constraint on HorizontalNav item count is available horizontal space, not a fixed number — long labels mean fewer items fit before the bar overflows.

- Pair with `renderLink` to integrate a framework router (Next.js, React Router) so navigation stays client-side.

- Keep `value`/`defaultValue` in sync with the current route so the active item reflects where the user is.

## Examples

### Basic Horizontal Nav

Use for top-level site navigation where users move between major sections.

```tsx
<HorizontalNav
  defaultValue="home"
  items={[
    { value: "home", href: "/", children: "Home" },
    { value: "about", href: "/about", children: "About" },
    { value: "contact", href: "/contact", children: "Contact" },
  ]}
/>
```

### With Icons Horizontal Nav

Use when icons help users quickly identify navigation destinations.

```tsx
<HorizontalNav
  defaultValue="home"
  items={[
    {
      value: "home",
      href: "/",
      children: (
        <>
          <Icon /> Home
        </>
      ),
    },
    {
      value: "about",
      href: "/about",
      children: (
        <>
          <Icon /> About
        </>
      ),
    },
  ]}
/>
```

### With Button Item Horizontal Nav

Use `asChild: true` on an item to render a non-anchor element (e.g. a button that opens a modal) while keeping it in the tab order.

```tsx
<HorizontalNav
  defaultValue="home"
  items={[
    { value: "home", href: "/", children: "Home" },
    { value: "about", href: "/about", children: "About" },
    {
      value: "settings",
      asChild: true,
      children: <button type="button">Settings</button>,
    },
  ]}
/>
```

### With Custom Link Horizontal Nav

Use `renderLink` to swap the underlying anchor for a framework router component (Next.js `Link`, React Router `Link`, etc.) so navigation stays client-side.

```tsx
<HorizontalNav
  defaultValue="home"
  items={[
    { value: "home", href: "/", children: "Home" },
    { value: "about", href: "/about", children: "About" },
  ]}
  renderLink={(item) => <Link {...item} />}
/>
```

### With Disabled Item Horizontal Nav

Use when a nav destination exists but is temporarily unavailable.

```tsx
<HorizontalNav
  defaultValue="home"
  items={[
    { value: "home", href: "/", children: "Home" },
    {
      value: "settings",
      href: "/settings",
      children: "Settings",
      disabled: true,
    },
    { value: "about", href: "/about", children: "About" },
  ]}
/>
```

### Composed

```tsx
<HorizontalNavRoot defaultValue="home">
  <HorizontalNavList>
    <HorizontalNavLink value="home" href="/">
      Home
    </HorizontalNavLink>
    <HorizontalNavLink value="about" href="/about">
      About
    </HorizontalNavLink>
    <HorizontalNavLink value="contact" href="/contact">
      Contact
    </HorizontalNavLink>
  </HorizontalNavList>
</HorizontalNavRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `HorizontalNavLinkProps[]` | - | Navigation items rendered in order across the bar. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ HorizontalNavList?: NativeElementAttributes<"div", typeof HorizontalNavList>; HorizontalNavLink?: NativeElementAttributes<"a", typeof HorizontalNavLink> }` | - | Native HTML attributes forwarded to the internal composed components. |
| defaultValue | `string` | - | Selected item's value on first render when the nav is uncontrolled. |
| onValueChange | `(value: string) => void` | - | Called with the new value when the selected item changes. |
| renderLink | `(item: Omit<HorizontalNavItem, "href"> & { href: string; }) => ReactNode` | - | Renders each link with a custom component, typically a framework-aware Link. Receives the item and returns the element to render. |
| value | `string` | - | Currently selected item's value when the nav is controlled. Pair with `onValueChange`. |

`* = required prop`

## Exports

- Components: `HorizontalNav`, `HorizontalNavLink`, `HorizontalNavList`, `HorizontalNavRoot`
- Types: `HorizontalNavItem`, `HorizontalNavLinkProps`, `HorizontalNavListProps`, `HorizontalNavProps`, `HorizontalNavRenderLinkItem`, `HorizontalNavRootProps`

## See also

- [VerticalNav](./VerticalNav.md)
- [AppBar](./AppBar.md)
