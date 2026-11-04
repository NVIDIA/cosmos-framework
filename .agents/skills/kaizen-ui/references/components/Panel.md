# Panel

Container that groups related content and actions with a background, border, and consistent padding. Use for general-purpose surfaces on dashboards and detail pages.

## Notes

- Panel manages its own internal padding. Do not add Tailwind padding classes (`p-4`, `p-5`, etc.) to `className` — they stack with the internal padding and create visual doubling.

- Panel represents "a region of content", Card represents "a thing": Card adds identity/interaction states (`interactive`, `selected`) and entity-specific slots (media, header). Use Panel for general-purpose groupings on dashboards and detail pages; use Card for surfaces representing a single subject.

- Anti-pattern: do not wrap primary tables on list-view pages in a Panel (render directly in `<main>`), and do not wrap tab content in a Panel on detail pages. Panel-wrapping tables is a dashboard pattern, not a generic rule.

- Elevation levels are `low`/`mid`/`high`/`higher`. In light mode only `low` changes the background — `mid`/`high`/`higher` render identically — so always test elevation in both themes.

- Default a standalone panel's background to `elevation="high"` (raised) — this matches the design library's visual default, even though the component's own prop default is `"mid"` for backward compatibility. `elevation` only changes the panel's background color, not a drop shadow. Reserve `"higher"` for a panel nested on top of another panel that needs more background contrast than `"high"` gives it, and use `"low"` for sunken or recessed content.

- When a Panel must contain another Panel, the inner Panel uses a higher elevation than the outer one; otherwise prefer headings and spacing over nesting.

- `slotFooter` is an action zone — use it for buttons that operate on the panel's content. Do not place metadata or informational text there; put it in the panel body instead.

## Examples

### Basic Panel

Use to wrap related content and actions in a card-like container with a background, border, and padding.

```tsx
<Panel slotHeading={<h2>Panel Title</h2>}>Panel content goes here.</Panel>
```

### With Icon Panel

Use when an icon helps identify the panel's purpose at a glance.

```tsx
<Panel slotHeading={<h2>Configuration</h2>} slotIcon={<Icon />}>
  Manage your settings and preferences.
</Panel>
```

### With Footer Panel

Use when the panel has actions that apply to its content, such as save or cancel buttons.

```tsx
<Panel
  slotHeading={<h2>Job Details</h2>}
  slotIcon={<Icon />}
  slotFooter={
    <>
      <Button kind="secondary">Cancel</Button>
      <Button color="brand">Save</Button>
    </>
  }
>
  Fill in the details for your new job.
</Panel>
```

### High Elevation Panel

Use when you need increased contrast between the panel and the page background, such as in layered UIs.

```tsx
<Panel slotHeading={<h2>Overlay Panel</h2>} elevation="high">
  This panel uses a raised background for increased contrast.
</Panel>
```

### Compact Density Panel

Use to override panel spacing locally when a single panel needs to be denser than the surrounding ThemeProvider density.

```tsx
<Panel density="compact" slotHeading={<h2>Storage</h2>} slotIcon={<Icon />} slotFooter={<Button color="brand">Save</Button>}>
  Tightens internal padding and footer spacing for dense layouts.
</Panel>
```

### Content Only Panel

Use for simple content containers where a heading is unnecessary.

```tsx
<Panel>Content without a heading or footer.</Panel>
```

### Composed

```tsx
<PanelRoot elevation="mid">
  <PanelHeader>
    <PanelIcon>
      <Icon />
    </PanelIcon>
    <PanelHeading>Panel Title</PanelHeading>
  </PanelHeader>
  <PanelContent>Panel content goes here.</PanelContent>
  <PanelFooter>
    <Button color="brand">Action</Button>
  </PanelFooter>
</PanelRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ PanelHeader?: NativeElementAttributes<"div", typeof PanelHeader>; PanelIcon?: NativeElementAttributes<"div", typeof PanelIcon>; PanelHeading?: NativeElementAttributes<"div", typeof PanelHeading>; PanelContent?: NativeElementAttributes<"div", typeof PanelContent>; PanelFooter?: NativeElementAttributes<"div", typeof PanelFooter> }` | - | Native HTML attributes forwarded to the internal composed components. |
| density | `"compact" \| "standard" \| "spacious"` | - | Density of the panel. |
| elevation | `"high" \| "low" \| "higher" \| "mid"` | `"mid"` | Surface elevation. Drives the background color used to indicate hierarchy against the page surface. |
| slotFooter | `ReactNode` | - | Footer content rendered below the body. Children are spaced automatically. |
| slotHeading | `ReactNode` | - | Heading rendered at the top of the panel, with `label/bold/xl` default typography styles |
| slotIcon | `ReactNode` | - | Icon rendered to the left of the heading. |

## Exports

- Components: `Panel`, `PanelContent`, `PanelFooter`, `PanelHeader`, `PanelHeading`, `PanelIcon`, `PanelRoot`
- Utilities: `panelRoot`
- Types: `PanelContentProps`, `PanelFooterProps`, `PanelHeaderProps`, `PanelHeadingProps`, `PanelIconProps`, `PanelProps`, `PanelRootProps`

## See also

- [Card](./Card.md)
