# TreeNav

A hierarchical navigation list that lets users move through links at different levels, similar to a directory tree

## Notes

- Set `active` on a leaf or branch trigger to highlight the current item. Branches with active descendants auto-expand.

- Set `disabled` on a leaf to make it non-interactive, or on a branch to disable the entire subtree — descendants inherit the disabled state.

- Set `loading` on an item to replace its icon with a spinner — typically combined with `onOpenChange` or `onSelect` to drive async data fetches.

- Set `collapsible` to false to force a branch to remain open.

- Pass `slotIcon` on any item to override the default document/folder icon.

## Examples

### Basic Tree Nav

Use for hierarchical navigation structures like file trees or nested categories.

```tsx
<TreeNav
  items={[
    {
      id: "src",
      children: "src",
      defaultOpen: true,
      items: [
        { id: "index", children: "index.ts", href: "#index" },
        { id: "utils", children: "utils.ts", href: "#utils" },
        {
          id: "input",
          asChild: true,
          children: <button type="button">Input.tsx</button>,
        },
      ],
    },
    { id: "readme", children: "README.md", href: "#readme" },
  ]}
/>
```

### Controlled Tree Nav

Pass both `open` and `onOpenChange` to drive branch expansion from external state (Redux, URL, parent component, etc.) instead of letting the component manage it.

```tsx
<TreeNav
  items={[
    {
      id: "src",
      children: "src",
      open: true,
      onOpenChange: handleOpenChange,
      items: [{ id: "app", children: "App.tsx", href: "#app" }],
    },
    {
      id: "lib",
      children: "lib",
      open: false,
      onOpenChange: handleOpenChange,
      items: [{ id: "utils", children: "utils.ts", href: "#utils" }],
    },
  ]}
/>
```

### With Render Link Tree Nav

Use `renderLink` to swap the built-in `<a>` for a framework router link (Next.js `<Link>`, React Router `<NavLink>`, TanStack Router, etc.).

```tsx
<TreeNav
  items={[
    {
      id: "pages",
      children: "pages",
      defaultOpen: true,
      items: [
        { id: "home", children: "Home", href: "/home" },
        { id: "about", children: "About", href: "/about" },
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

### Composed

```tsx
<TreeNavRoot>
  <TreeNavList>
    <TreeNavBranch defaultOpen>
      <TreeNavBranchTrigger>src</TreeNavBranchTrigger>
      <TreeNavList>
        <TreeNavLeaf href="#index">index.ts</TreeNavLeaf>
        <TreeNavBranch defaultOpen>
          <TreeNavBranchTrigger>components</TreeNavBranchTrigger>
          <TreeNavList>
            <TreeNavLeaf href="#button" active>
              Button.tsx
            </TreeNavLeaf>
            <TreeNavLeaf active asChild>
              <button type="button">Input.tsx</button>
            </TreeNavLeaf>
          </TreeNavList>
        </TreeNavBranch>
      </TreeNavList>
    </TreeNavBranch>
    <TreeNavLeaf href="#readme">README.md</TreeNavLeaf>
  </TreeNavList>
</TreeNavRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `TreeNavBaseItem[]` | - | Tree definitions rendered in order. |
| renderLink | `(item: Omit<TreeNavBaseItem, "href"> & { href: string; }) => ReactNode` | - | Renders a custom element in place of the default `<a>` for each leaf, useful for framework-specific link components. |

`* = required prop`

## Exports

- Components: `TreeNav`, `TreeNavBranch`, `TreeNavBranchTrigger`, `TreeNavIcon`, `TreeNavLeaf`, `TreeNavList`, `TreeNavRoot`
- Types: `TreeNavBaseItem`, `TreeNavBranchProps`, `TreeNavBranchTriggerProps`, `TreeNavIconProps`, `TreeNavLeafProps`, `TreeNavListProps`, `TreeNavProps`, `TreeNavRenderLinkItem`, `TreeNavRootProps`
