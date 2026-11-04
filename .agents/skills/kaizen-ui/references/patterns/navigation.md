# Navigation

Navigation is never a single component choice. Each surface answers a different wayfinding question, and most pages need several at once. Don't ask "which nav component?" — ask "which questions must this page answer?", then pick a surface for each.

## When to use

| The user is asking… | Scope | Surface |
| --- | --- | --- |
| **Where can I go?** (what areas exist) | App | Horizontal app navigation, vertical section navigation, or both (see [Variants](#variants)) |
| **Where am I?** (within the current area) | Section | Vertical section navigation or in-page tabs |
| **How did I get here — and back?** | Path | Breadcrumbs |
| **What's on this page?** (how its content is organized) | Content | Tabs or accordions in the page body |

These surfaces **layer, they don't compete:**

- A typical detail page answers three at once — app nav (where I can go), breadcrumbs (how I got here), tabs (what's on the page).
- **Never answer the same question twice.** If breadcrumbs show the path, the sidebar shouldn't re-encode it; if tabs make position obvious, you may not need breadcrumbs (see [Breadcrumbs](#breadcrumbs) → "Not needed when").
- **Decide the app-level surface first** (see [Variants](#variants)). It constrains content width and which other patterns stay viable.

---

## Variants

Choose the primary nav pattern based on top-level destination count, section depth, and label complexity. Content-level variants (tabs, accordions) compose with any primary nav.

### Horizontal Navigation (Default)

Use [`AppBar`](../components/AppBar.md) with [HorizontalNav](../components/HorizontalNav.md).

Use when ALL of the following are true:

- The application has fewer than 7 top-level destinations
- No top-level destination requires its own section-level sub-navigation
- Destinations are self-evident from their labels

> **KUI convention note:** The threshold of 7 is a NVIDIA convention reflecting typical label lengths in NVIDIA tools and the width of AppBar's center slot. Real capacity depends on label length and available width, not a fixed count — short labels may fit 7–8, long labels fewer.

This preserves full page width for data-dense views such as tables and dashboards. Put the horizontal navigation in the AppBar's documented navigation region. See [`page-shell.md`](../layouts/page-shell.md#stacked-shell).

Adding vertical navigation "for future expansion" is an antipattern. A detail page with 5–6 items does not justify a sidebar; promote only when the rules above are met.

### Horizontal + Vertical Navigation (Global + Local)

Use [`AppBar`](../components/AppBar.md) with [HorizontalNav](../components/HorizontalNav.md) and [VerticalNav](../components/VerticalNav.md).

Use when ALL of the following are true:

- The application has distinct top-level areas shown horizontally
- At least one top-level area has enough internal structure to need its own
  local navigation (e.g., Documentation → Get Started, Guides, Components)
- The vertical nav is scoped to the selected top-level area

This pattern separates global wayfinding (horizontal) from local wayfinding (vertical). Keep the two visually and spatially distinct so each level occupies a consistent, different location.

The horizontal surface carries global areas; the vertical surface is **scoped to the selected top-level area** and changes per section. Keep each level in a stable, distinct location.

For the surrounding shell (sidebar grid, sticky aside offset by the AppBar height, mobile drawer behavior) see [`page-shell.md`](../layouts/page-shell.md#fixed-sidebar-shell).

### Vertical Navigation Alone

Use [VerticalNav](../components/VerticalNav.md).

Use when ANY of the following are true:

- The entire application is a single product area with 7+ destinations
  and/or multi-level hierarchy
- No meaningful top-level split exists to justify horizontal separation
- All destinations belong to one domain and users navigate between them
  frequently

### Breadcrumbs

Component: [`Breadcrumbs`](../components/Breadcrumbs.md).

Use breadcrumbs when ALL of the following are true:

- The user navigates 3 or more levels deep in the application hierarchy
- No other wayfinding mechanism (tabs, visible navigation trail) makes the user's current position obvious
- The user may need to jump to an intermediate level

**Not needed when:**

- Tabs are present that show the user's current position within a section (tabs already communicate the trail)
- The application has only 1–2 levels of depth
- The navigation sidebar already shows the user's full path via expanded / highlighted states

### Tabs

Use [Tabs](../components/Tabs.md).

Use when:

- Content on a single page can be divided into 2–7 parallel categories
- The user benefits from switching between views without leaving the page
- Each tab represents a complete, independent view of content (not sequential steps — use a stepper/wizard for that)
- The categories are peer-level — no tab is subordinate to another

**Placement:**

- Directly below the page header or section header they relate to
- Never nested more than one level deep (tabs within tabs creates confusion — use a different pattern for the second level)

If tabs are part of the URL structure (each tab has its own route), they function as lightweight navigation. If they are not part of the URL, they are purely content organization.

### Accordions / Collapsible Sections

Use [`Accordion`](../components/Accordion.md) or [Collapsible](../components/Collapsible.md).

Use for progressive disclosure when a single page is long and users need only one or two sections at a time. Don't use to hide primary content the user came for.

---

## Composition rules

Apply to all variants.

### What belongs as a primary nav item

Top-level navigation items are **destinations** — places users go to orient, browse, or monitor. They are not actions on specific entities.

**Navigation destinations (belong in top-level nav):**

- **Collection views:** "Browse clusters", "My requests", "Request queue"
- **Dashboards:** "Utilization", "Overview"
- **Settings or admin areas:** "Settings", "Team management"

**NOT navigation destinations (do not put in top-level nav):**

- **Actions on an entity:** "Configure cluster", "Edit request", "Create task"
- **Forms reached from entity context:** creating, editing, or configuring a specific item is an action launched from the entity's row, detail page, or a contextual button — not a peer of browse/dashboard destinations

**Where entity actions live instead:**

- **Row action** in a browse table (button, link, or row click)
- **Action button** in the detail page header
- **"New [entity]" button** in a list view toolbar (for create flows)
- **Contextual link** from a related entity's detail page

Entity actions in top-level nav break the mental model (nav = where I go, actions = what I do to something) and appear selectable without context. **If the user's prompt or feature spec places an action in the nav,** flag it and suggest an alternative entry point rather than silently adding it.

### Role-gated items: hide, don't disable

When a user's role does not grant access to a destination, **hide the item entirely** — disabled nav items create false affordance and leak the existence of restricted areas. This is [element visibility](./element-visibility.md) → Question 1 applied to nav.

- For a restricted URL hit directly (bookmark, shared link), show a permission-denied state on the page itself (see `../layouts/list-view.md` → Permission denied empty state), not a disabled nav item.
- **Exception:** a preview/beta feature the user can request access to may show a "Request access" affordance — a product decision, not a default.

## Responsive Adaptation

Every navigation decision should include a responsive plan.

- Horizontal navigation — keep visible while it fits; collapse to a labeled menu when overflow becomes likely.
- Vertical navigation — move into an accessible drawer on narrow viewports.
- Tabs — preserve the active tab and use the target framework's documented overflow behavior.
