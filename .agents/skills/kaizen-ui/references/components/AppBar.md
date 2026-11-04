# AppBar

Also known as: Masthead, Navbar, Navigation Bar

The top-level masthead for an application, anchoring branding, primary navigation, and global actions. Every application has exactly one.

## Notes

- Render exactly one AppBar per application.

- Keep `slotEnd` to two or three global actions; group overflow inside a Dropdown rather than adding more icons.

- Place version or product switchers in `slotStart` next to the logo as a Dropdown whose trigger shows the current selection.

- `slotStart` and `slotEnd` already render as `flex items-center` with the AppBar's `gap` inherited; `slotStart` also applies the masthead label style. Pass children directly as a Fragment (`<>...</>`) — don't wrap them in your own `flex`/`gap` container, and don't re-style the product label with a styled `<span>`; plain span or Anchor with textKind="inherit" inherits the correct style.

- Implement AppBar as a sticky top masthead (e.g. `sticky top-0` plus a stacking `z-index`) so brand, nav, and global actions stay visible while the body scrolls.

- When a sticky VerticalNav sidebar is used alongside AppBar, set its `top-*` offset to match the AppBar block height (via measured height, design token(--nv-app-bar-height), or `calc`) so the sidebar fills the viewport below the masthead instead of sliding under it.

- Anti-patterns: non-sticky AppBar on long pages, navigation built outside the AppBar (custom header), too many actions in the trailing slot

## Examples

### Simple Example

```tsx
<AppBar
  slotStart={
    <>
      <AppBarExpanderButton />
      <Anchor kind="standalone" href="/" textKind="inherit">
        Product
      </Anchor>
    </>
  }
/>
```

### Full Example

A full example of the AppBar component with a start and end slot. Note the <span>ACME</span> is a placeholder for a brand SVG or logo.

```tsx
<AppBar
  slotStart={
    <>
      <AppBarExpanderButton onClick={() => console.log("toggle")} />
      <span className="font-bold">ACME</span>
      <Anchor kind="standalone" href="/" textKind="inherit">
        Product Name
      </Anchor>
    </>
  }
  slotEnd={<Avatar interactive fallback="AC" />}
>
  <HorizontalNav
    defaultValue="home"
    items={[
      { value: "home", href: "/", children: "Home" },
      { value: "about", href: "/about", children: "About" },
      { value: "contact", href: "/contact", children: "Contact" },
    ]}
  />
</AppBar>
```

### With Centered Content

```tsx
<AppBar>
  <div className="mx-auto">
    <HorizontalNav
      defaultValue="home"
      items={[
        { value: "home", href: "/", children: "Home" },
        { value: "about", href: "/about", children: "About" },
        { value: "contact", href: "/contact", children: "Contact" },
      ]}
    />
  </div>
</AppBar>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| children | `ReactNode` | - | Center-slot content, left-aligned by default and typically used for primary navigation. |
| slotEnd | `ReactNode` | - | Trailing-edge content, typically global actions such as a user menu or settings. |
| slotStart | `ReactNode` | - | Leading-edge content, typically brand area and product name. |

## Exports

- Components: `AppBar`, `AppBarExpanderButton`
- Constants: `AppBarHeight`, `AppBarHeightVariable`
- Types: `AppBarProps`

## See also

- [HorizontalNav](./HorizontalNav.md)
- [VerticalNav](./VerticalNav.md)
