# Layout Principles

Foundational rules that apply to every page layout. Reference these when making spacing, containment, and structural decisions.

**Determine the navigation pattern before laying out the page.** Navigation constrains content width and page structure. See [navigation.md](../patterns/navigation.md) to decide between only horizontal nav vs vertical nav vs both. Vertical nav is an **application** decision, not a page decision — a single detail, browse, or form page does not justify adding a sidebar.

Every page renders inside a shell - see [page-shell.md](./page-shell.md) to ensure you use the correct shell.

See [card grid layout](../patterns/entity-cards.md#layout) when laying out cards.

---

## Spacing hierarchy

Spacing communicates relationship. Closer elements are more related. There are three tiers:

| Tier          | Size    | Use for                                                                                                    |
| ------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| **Tight**     | 4-8px   | Elements that form a single control (label + input, icon + text). Perceived as a single unit.              |
| **Related**   | 12-16px | Siblings within a section (fields in a group, items in a list). Belong together but individually distinct. |
| **Sectional** | 24-32px | Distinct regions or topic changes. Separate regions of different purpose.                                  |

**Rules:**

- **Do not use in-between values.** If you find yourself reaching for 20px, decide whether the relationship is _related_ (16px) or _sectional_ (24px). In-between values signal indecision.
- **Do not exceed 32px between any two elements on a page.** Gaps larger than 32px read as missing content, not breathing room. If a section needs more separation, use a subtle divider or background change.
- Horizontal and vertical spacing follow the same tiers.

### Common Spacing Gaps

| Transition                         | Value | Rationale                                         |
| ---------------------------------- | ----- | ------------------------------------------------- |
| Page header → first content region | 32px  | Sectional separation between header and body      |
| Filter bar → results region        | 24px  | Filters and results are conceptually linked       |
| Section → section (peer weight)    | 32px  | Sibling sections of similar importance            |
| Tab bar → tab content              | 24px  | Tab content is a child region, not a peer section |
| Two-column horizontal gap          | 24px  | Columns read as peers, not discrete sections      |
| Page content → footer              | 32px  | Sectional release at page end                     |

**Hero-led pages are an exception** — hero → content transitions run at 64px because the hero is a visual reset, not a peer section.

---

## Containment

Containers (panels, cards, bordered regions) signal that contents are a unified, interactive group. They add visual weight — use them only when that weight solves a problem spacing cannot.

**Core principle:** containers resolve ambiguity that layout and spacing cannot. Before adding a container, ask: _Is there ambiguity about what belongs together here?_ If columns, tabs, or spacing already create clear grouping, a container adds noise.

### Use a container when

1. **Interaction scope must be explicit** — content is collapsible, dismissible, or selectable as a unit
2. **Sibling objects compete for attention** — cards in a grid, repeated panels in a region
3. **Visual conflict exists** — sticky/floating content overlays other content
4. **Region is an authoring zone** — comment boxes, form sections, editors

### Skip the container when

1. **Layout provides boundaries** — two-column structure, tabs already separate regions
2. **Content flows sequentially** — documentation, timeline, linear narrative
3. **Container would be full-width with no siblings** — a panel that spans the page adds no grouping value

### Containment strength

| Ambiguity level                    | Solution                            |
| ---------------------------------- | ----------------------------------- |
| None (own column/region)           | No container — heading + spacing    |
| Low (sequential sections)          | Divider or subtle background accent |
| Medium (sibling objects)           | Card or panel                       |
| High (overlapping z-index, sticky) | Container + elevation               |

### Dividers vs. containers

| Element                             | Use when                                                                           |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| **Dividers** (borders, rules)       | Separating adjacent static regions in the same plane. Neither side is an "object." |
| **Containers** (backgrounds, cards) | Wrapping content that is an object — selectable, movable, collapsible, floating.   |

### When NOT to use dividers

- **Between key-value pairs:** spacing alone is sufficient. Dividers between every field create visual noise.
- **Inside status bars:** spacing separates metrics; dividers add noise.
- **Between sections with headings:** the heading already signals a new section. Add sectional spacing (24px), not a divider.
- **Above or below a single element:** a divider needs content on both sides. Don't use it as decoration.

### Over-containment signals

- More than 4-5 horizontal dividers in a single viewport → over-segmented
- Nested containers → flatten and use spacing/headings instead
- Full-width borders with no siblings → false floor that stops visual scanning

### Containers and spacing are independent

Containment does not substitute for spacing. A card still follows the spacing hierarchy — elements inside use tight/related spacing, and the card itself sits at sectional distance from adjacent content.

---

## Multi-column layouts

### When to use two columns

Use 60/40 or 50/50 two-column when:

- Content naturally divides into distinct concerns (what it does vs. how it's configured)
- Both columns have enough content to justify the split
- Both columns fit above the fold on typical screens (1280px+)

### When to use single column

Use single column when:

- Content is sequential or narrative
- One content type dominates (long form, visualization)
- Screen width is constrained (<900px)
- Form steps where focus should be linear

### Column assignment heuristic

| Content type                    | Column            |
| ------------------------------- | ----------------- |
| Primary purpose / core function | Left (primary)    |
| Supporting context / metadata   | Right (secondary) |
| Interactive elements            | Usually left      |
| Reference information           | Usually right     |

---

## Section ordering

Within a page or column, order content by:

1. **Frequency of access** — most-checked information first
2. **Dependency** — information that informs decisions comes before those decisions
3. **Stability** — frequently-changing values above rarely-changing ones
4. **Task flow** — match the order the user thinks about the task

---

## Density

Density should match the user's task and expertise level.

| Level        | Spacing            | Use for                                                                     |
| ------------ | ------------------ | --------------------------------------------------------------------------- |
| **Compact**  | Lower end of tiers | Expert users, known tasks: tables, data grids, developer tools              |
| **Medium**   | Standard           | Mixed audiences, varied tasks: catalog browsing, search results, list views |
| **Spacious** | Upper end of tiers | New users, focused tasks: onboarding, setup wizards, landing pages          |

**Rules:**

- Read the target audience to determine density — experienced practitioners default to medium-to-compact
- Don't mix density levels within a single view unless a clear spatial boundary exists (sidebar vs. main content)
