# Page Shell

Every page starts with an [AppBar](../components/AppBar.md) spanning the viewport and a padded, width-constrained native `main` below it. Content must not touch viewport edges. Always include a native home link with the product name.

## Stacked Shell

Default to an AppBar with horizontal application navigation stacked above the content.

Use for all browse, detail, and form pages unless the app requires a sidebar.

```tsx
<div className="min-h-screen bg-surface-base">
  <header className="sticky top-0 z-10">
    <AppBar
      slotStart={
        <>
          <AppBarExpanderButton className="sm:hidden" />
          <span className="font-bold">ACME</span>
          <Anchor kind="standalone" href="/" textKind="inherit">
            App Name
          </Anchor>
        </>
      }
    >
      <HorizontalNav className="hidden sm:flex" items={globalNavItems} />
    </AppBar>
  </header>
  <main className="mx-auto w-full max-w-7xl px-[clamp(16px,5vw,32px)] py-6">
    {/* children - rest of app */}
    <AppShellChildren />
  </main>
</div>
```

## Fixed Sidebar Shell

Use a fixed sidebar only when the app has 7+ destinations or nested section navigation. Keep horizontal global navigation only when it answers a separate wayfinding question.

- **Mobile:** prefer a drawer behavior for the vertical nav. If the sidebar stacks above content, keep `overflow-x: hidden` on the aside.

```tsx
<div className="grid h-screen grid-rows-[auto_1fr]">
  <header>
    <AppBar
      slotStart={
        <>
          <AppBarExpanderButton />
          <span className="font-bold">ACME</span>
          <Anchor kind="standalone" href="/" textKind="inherit">
            App Name
          </Anchor>
        </>
      }
    >
      <HorizontalNav className="hidden sm:flex" items={globalNavItems} />
    </AppBar>
  </header>

  <div className="grid min-h-0 grid-cols-[auto_1fr]">
    <aside>
      <VerticalNav className="hidden md:flex" items={sectionNavItems} />
    </aside>

    <main className="mx-auto min-h-0 w-full max-w-7xl overflow-auto px-[clamp(16px,5vw,32px)] py-6">
      <AppShellChildren />
    </main>
  </div>
</div>
```

## Collapsible Sidebar Shell

Let the AppBar expander collapse the sidebar into a 56px icon rail when pages need their full width. The content keeps the remaining width, and the rail keeps every destination one click away. Give every top-level destination an icon; its label stays the accessible name.

- **Push:** the sidebar stays in the page flow beside the content. Toggle the rail from the expander and reflect the state in its `aria-expanded`.
- **Hidden:** remove the sidebar instead of collapsing it when the page needs every pixel.
- **Overlay and mobile:** below the sidebar breakpoint, open the nav above the content in a SidePanel from the same expander, so it gets a blanket, a focus trap, and Escape to close.

```tsx
() => {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <div className="grid h-screen grid-rows-[auto_1fr]">
      <header>
        <AppBar
          slotStart={
            <>
              <AppBarExpanderButton aria-expanded={!collapsed} aria-label="Toggle navigation" onClick={() => setCollapsed((value) => !value)} />
              <span className="font-bold">ACME</span>
              <Anchor kind="standalone" href="/" textKind="inherit">
                App Name
              </Anchor>
            </>
          }
        />
      </header>

      <div className="grid min-h-0 grid-cols-[auto_1fr]">
        <aside className="overflow-x-hidden">
          <VerticalNav collapsed={collapsed} aria-label="Section" items={sectionNavItems} />
        </aside>

        <main className="mx-auto min-h-0 w-full max-w-7xl overflow-auto px-[clamp(16px,5vw,32px)] py-6">
          <AppShellChildren />
        </main>
      </div>
    </div>
  );
};
```

## Shell Rules

| Rule | CSS requirement | Rationale |
| --- | --- | --- |
| **Top of page AppBar** | Sticky at `top: 0` with a stacking level above page content | Global chrome stays visible while content scrolls. |
| **Responsive horizontal gutters** | `padding-inline: clamp(16px, 5vw, 32px)` | Based on screen width |
| **Vertical breathing room** | `padding-block: 1.5rem` | Space page content and edges |
| **Max-width + centering** | Full width, centered, with the page-type max-width below | Constrain content; prevent ultra-wide stretching |

## Max-width by page type

| Page type   | Max-width  |
| --- | --- |
| Browse (table or cards) | 80rem (1280px) |
| Detail (two-column)     | 80rem (1280px) |
| Form (single column)    | 56rem (896px); use 48rem (768px) for simple forms with fewer than 6 fields |

**Non-negotiable:** the `<main>` wrapper with responsive padding is required on every page. If page content touches the viewport edge, the shell is missing.

**Layout examples:** page-shell examples include the full shell structure (`AppBar` + `<main>`). For inner-page composition examples, avoid adding nested `<main>` wrappers. Apply the CSS requirements above through the application's configured utility classes or stylesheet.

**Full-bleed exception:** only hero bands, like on a landing page. Default body content always keeps gutters.

**Responsive two-column body:** two-column content must stack on narrow viewports
