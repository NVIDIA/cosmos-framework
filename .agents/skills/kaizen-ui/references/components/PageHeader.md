# PageHeader

Top-of-page banner that establishes context with a title, breadcrumbs, and page-level actions.

## Notes

- Do not place back-navigation buttons ("Back to [list]") in `slotActions`. Back navigation belongs in `slotBreadcrumbs` via the Breadcrumbs component.

- Use `kind="flat"` when the header sits flush with the page content (list, browse, dashboard). Reach for `kind="floating"` whenever the header needs visual separation from the content below — typically detail pages, but also any surface where a card-like appearance helps the header stand apart.

- Always provide `slotHeading`.

- Keep `slotActions` to one primary action and at most two more (overflow goes in a Dropdown). The primary action must use `color="brand"`; bare `kind="primary"` renders neutral and does not signal the page CTA.

- For entities with a status (Active, Failed, Running, etc.), render the Badge inline inside `slotHeading` (Flex with the title), not in a separate metadata band below the header.

- Keep `slotDescription` to one or two sentences. Move longer content into the page body.

- Reach for PageHeader on standard application pages (list, detail, form, dashboard, settings). Use Hero only for marketing or onboarding pages where visual impact is the goal; never render both.

- Decide PageHeader vs Hero by page intent: task-oriented pages (configure, monitor, edit, browse) use PageHeader; pages that orient or motivate (landing, marketing, onboarding) use Hero. AppBar coexists with either, and a page should have one PageHeader OR one Hero, not both.

- Overflow beyond 3 actions goes in a Dropdown ("More actions" / kebab); the primary action should be the most common next step ("Create", "Deploy", "Add"); if no action is clearly primary, render all secondary buttons rather than forcing one.

- Include `slotBreadcrumbs` whenever the page sits deeper than one level in the hierarchy.

- Omit `slotSubheading` when the breadcrumb's last segment already names the entity type (e.g. breadcrumb `GPU clusters > orion-prod` plus heading `orion-prod` makes `slotSubheading="GPU cluster"` redundant). Use it only when the breadcrumb is absent or generic, or to surface a meaningful parent not shown in the trail.

- Put non-identity metadata like "Created" or "Last updated" in `slotDescription` or a smaller metadata row after the heading; keep status with the name as an inline Badge and keep timestamps separate.

## Examples

### Basic Page Header

Use at the top of a page to describe its purpose with a title and optional description.

```tsx
<PageHeader slotSubheading="Subheading" slotHeading={<h1>Page Title</h1>} slotDescription="A brief description of what this page contains." />
```

### Floating Page Header

Use on detail pages where the header should read as a card and stand apart from page content.

```tsx
<PageHeader kind="floating" slotSubheading="Subheading" slotHeading={<h1>Page Title</h1>} slotDescription="A brief description of what this page contains." />
```

### With Breadcrumbs Page Header

Use for back-navigation and location context on nested pages. Back navigation belongs here, never in slotActions.

```tsx
<PageHeader
  slotBreadcrumbs={<Breadcrumbs items={[{ children: <a href="/">Home</a> }, { children: <a href="/models">Models</a> }, "Llama 3.1 405B"]} />}
  slotSubheading="NVIDIA"
  slotHeading={<h1>Llama 3.1 405B</h1>}
  slotDescription="A large language model optimized for dialogue and instruction-following tasks."
/>
```

### With Actions Page Header

Use when the page has primary actions like create, deploy, or export that belong in the header.

```tsx
<PageHeader
  slotHeading={<h1>Clusters</h1>}
  slotDescription="Manage your compute clusters and infrastructure."
  slotActions={
    <>
      <Button kind="secondary">Export</Button>
      <Button color="brand">Create Cluster</Button>
    </>
  }
/>
```

### With Non Stretching Actions Page Header

Wrap actions in a Flex container when the buttons should keep their natural size instead of stretching to fill the footer on narrow viewports.

```tsx
<PageHeader
  slotHeading={<h1>Page Title</h1>}
  slotSubheading="Subheading"
  slotDescription="A brief description of what this page contains."
  slotActions={
    <Flex gap="density-md" align="end">
      <Button kind="secondary">Secondary</Button>
      <Button color="brand">Primary</Button>
    </Flex>
  }
/>
```

### With Custom Heading Page Header

Pass JSX into slotHeading when the title needs an inline adornment such as a status badge or a custom heading element.

```tsx
<PageHeader
  slotSubheading="Subheading"
  slotHeading={
    <Flex gap="density-md" align="center">
      <h1>Page Title</h1>
      <Badge>Coming soon</Badge>
    </Flex>
  }
  slotDescription="A brief description of what this page contains."
>
  <Badge kind="solid">Language</Badge>
</PageHeader>
```

### Composed

```tsx
<PageHeaderRoot>
  <PageHeaderContainer kind="floating">
    <PageHeaderContent>
      <PageHeaderHeader>
        <PageHeaderSubheading>Subheading</PageHeaderSubheading>
        <PageHeaderHeading>Page Title</PageHeaderHeading>
        <PageHeaderDescription>A brief description of what this page contains.</PageHeaderDescription>
      </PageHeaderHeader>
      <PageHeaderMain>Additional content area</PageHeaderMain>
    </PageHeaderContent>
    <PageHeaderFooter>
      <Button color="brand">Action</Button>
    </PageHeaderFooter>
  </PageHeaderContainer>
</PageHeaderRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| attributes | `{ PageHeaderContainer?: NativeElementAttributes<"div", typeof PageHeaderContainer>; PageHeaderContent?: NativeElementAttributes<"div", typeof PageHeaderContent>; PageHeaderHeader?: NativeElementAttributes<"div", typeof PageHeaderHeader>; PageHeaderSubheading?: NativeElementAttributes<"div", typeof PageHeaderSubheading>; PageHeaderHeading?: NativeElementAttributes<"div", typeof PageHeaderHeading>; PageHeaderDescription?: NativeElementAttributes<"div", typeof PageHeaderDescription>; PageHeaderMain?: NativeElementAttributes<"div", typeof PageHeaderMain>; PageHeaderFooter?: NativeElementAttributes<"div", typeof PageHeaderFooter> }` | - | Native HTML attributes forwarded to the internal composed components. |
| children | `ReactNode` | - | Additional content rendered below the description, such as tags, badges, or progress indicators. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces. |
| kind | `"flat" \| "floating"` | `"flat"` | Visual treatment of the page header. - "flat" - No background or padding. Use when the header sits flush with the content below (list, browse, dashboard). - "floating" - Bordered container with a background. Use whenever the header needs visual separation from the content below — typically detail pages, but also any surface where a card-like appearance helps the header stand apart. |
| slotActions | `ReactNode` | - | Page-level action buttons. Items stretch to fill the width on smaller screens. |
| slotBreadcrumbs | `ReactNode` | - | Breadcrumbs rendered above the heading area. |
| slotDescription | `ReactNode` | - | Description rendered below the heading. |
| slotHeading | `ReactNode` | - | Title of the page. |
| slotSubheading | `ReactNode` | - | Supporting text rendered above the heading. |

## Exports

- Components: `PageHeader`, `PageHeaderContainer`, `PageHeaderContent`, `PageHeaderDescription`, `PageHeaderFooter`, `PageHeaderHeader`, `PageHeaderHeading`, `PageHeaderMain`, `PageHeaderRoot`, `PageHeaderSubheading`
- Types: `PageHeaderContainerProps`, `PageHeaderContentProps`, `PageHeaderDescriptionProps`, `PageHeaderFooterProps`, `PageHeaderHeaderProps`, `PageHeaderHeadingProps`, `PageHeaderMainProps`, `PageHeaderProps`, `PageHeaderRootProps`, `PageHeaderSubheadingProps`

## See also

- [Breadcrumbs](./Breadcrumbs.md)
- [Button](./Button.md)
- [Badge](./Badge.md)
