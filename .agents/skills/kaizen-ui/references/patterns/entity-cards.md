# Entity Cards

The content anatomy of a summary card that represents a single entity inside a collection. Cards trade information density for scannability and visual differentiation.

For the `Card` component's exact API, slots, interaction model, and examples, see [`Card.md`](../components/Card.md). This file is about **what content goes inside** a card in a collection.

## When to use

Applies once you've decided cards are the right display method over tables or lists. See [`data-display.md`](./data-display.md) → "When to use cards vs. tables vs. lists".

---

## Anatomy

Every card in a collection shares the same slot structure. Variation comes from content values, not from which slots are present.

### Mapping content to Card regions

Map the pattern's content regions to the framework-specific slots documented in [`Card.md`](../components/Card.md):

| Pattern content | Card placement |
| --- | --- |
| Resource type | Header region — a single `Badge` |
| Visual element (thumbnail/hero) | Media region |
| Publisher → title → description → tags → footer stats | Content region |

Keep the type badge separate from the body content so it scans before the title. Use only slot names documented by the target framework's Card reference.

### Required slots

| Slot | Purpose |
| --- | --- |
| **Title** | Primary identifier. Prominent and scannable. |
| **Supporting text / description** | 1–2 lines summarizing what this item is. Browsing context — not the full detail. |

### Optional slots

Include when the content type warrants them. If one card in a collection includes a slot, every card in that collection must include it (use a placeholder over omission — see [Composition rules](#composition-rules)).

| Slot | Include when |
| --- | --- |
| **Visual element** | Content benefits from visual differentiation (thumbnails, icons, logos) — rendered in the media region |
| **Metadata** | Secondary attributes (date, size, count, author, duration) help the user evaluate or compare at browsing level |
| **Tags / labels** | Topic facets go in the body tags row; **resource type** (the categorical "what kind of thing is this") goes in the header as a `Badge` when kinds are mixed |
| **Ratings / reviews** | Content supports user feedback and ratings are meaningful for the browsing decision |
| **Action buttons** | User needs to take a direct action without navigating to the detail view |

### Canonical slot order

When a card includes multiple slots, arrange them in this order. Deviating signals either wrong content for a card or that a different card variant is needed.

| Position | Slot | Notes |
| --- | --- | --- |
| 1 | **Publisher / author / source** | **Always present.** Keep visible even when all cards share the same source — consistency outweighs the redundancy. Render small and muted so it doesn't compete with the title. **If your data model lacks this field, add one before rendering — do not drop the slot.** If genuinely unknown, render a muted placeholder ("Unknown" or the source system name). |
| 1a | **Resource type (when needed)** | When the grid mixes **resource kinds**, render one [`Badge`](../components/Badge.md) in the Card header so type scans before the title. **Operational state** belongs near the title on detail views; browse-card type is categorical, not operational state. |
| 2 | **Title** | Primary identifier. Largest text on the card. |
| 3 | **Supporting text / description** | 1–2 lines, truncated with ellipsis if needed. |
| 4 | **Topic / facet tags** | Inline row of [`Tag`](../components/Tag.md) components using the neutral gray outline treatment. Do **not** use this row for resource type; that belongs in the header Badge. |
| 5 | **Footer metadata (stats)** | Bottom row. 2–3 stats recommended (4 max before noise). Stat order is a product decision, but the chosen order must be **identical across every card in the collection**. |

**Why this order:**

- **Publisher-top** matches top-down scanning and lets users contextualize "who made this" before reading the title.
- **Tags above footer stats** keeps categorical info near the title (for filtering decisions) and quantitative info at the bottom (for cross-card comparison).
- **Publisher always visible** prevents a silent anatomy change when a filter narrows the collection to a single publisher.

**Visual rhythm:**

```text
┌─────────────────────────────────┐
│ [Type]                          │  ← header: Badge for resource type (when mixed kinds)
├─────────────────────────────────┤
│ {Publisher}                     │  ← content 1: publisher (muted)
│ {Title}                         │  ← content 2: title (dominant)
│ {1–2 line description}          │  ← content 3: description
│                                 │
│ [Topic] [Topic] [Topic]         │  ← content 4: topic tags (outline / gray)
│                                 │
│ {stat} · {stat} · {stat}        │  ← content 5: footer stats (2–3 items)
└─────────────────────────────────┘
```

### Slot typography

Use these kinds consistently across browse, landing, and detail-page related-item cards. Do not invent alternative mappings per page.

| Slot | Text kind | Color | Notes |
| --- | --- | --- | --- |
| 1 — Publisher | `label/regular/sm` | `text-secondary` | Muted. Sentence case — never uppercase |
| 1a — Type badge (header) | N/A (inside `Badge`) | Badge default / semantic | Single categorical type only; see [`Badge.md`](../components/Badge.md) |
| 2 — Title | `body/bold/xl` | `text-primary` (default) | Figma default is `body/bold/2xl`; `body/bold/xl` is the common NVIDIA usage. Do NOT use `heading/*` — that kind does not exist |
| 3 — Description | `body/regular/sm` | `text-secondary` | 1–2 lines; truncate with ellipsis |
| 4 — Topic tags | `label/regular/sm` (inside Tag) | Tag default (`outline`, `gray`) | Never per-value colors (rainbow coding). See [`Tag.md`](../components/Tag.md) |
| 5 — Footer metadata | `label/regular/sm` | `text-placeholder` | Browse card footer stats line only — scoped exception per [`COLORS.md`](../design-tokens/COLORS.md). Elsewhere metadata stays `text-secondary`. |

### When to deviate from the canonical order

| Case | Override |
| --- | --- |
| Card is visual-led (thumbnail dominates) | Thumbnail moves to the media region above the body; all other content stays in canonical order |
| Card is action-led (e.g., "Create new" tile) | Use a different card variant — this anatomy doesn't apply |

---

## Composition rules

Apply to all variants.

### Footer metadata row: left-aligned, never `justify="between"`

The footer (slot 5) reads as _one unit of information_, not three independent items. Use a left-aligned flex row with a fixed 24px gap or a dot-separator inline string. Left-alignment + fixed gap keeps items visually tethered; spreading items across the available width produces 100–150px gaps on wider cards that look like missing content.


```tsx
/* WRONG — justify="between" spreads 2–3 short items across the
    card's full width, creating 100–150px gaps on wider cards and
    breaking the "coherent metadata strip" reading pattern */
<Flex justify="between" align="center" gap="4">
  ...
</Flex>;

/* CORRECT — left-aligned Flex with fixed gap */
<Flex gap="6" align="center" wrap="wrap">
  ...
</Flex>;

/* ALSO CORRECT — dot-separated inline text */
<Text kind="label/regular/sm" className="text-placeholder">
  {stats.join(" · ")}
</Text>;
```







**Color:** use `text-placeholder` for the browse **card footer stats** strip (slot 5) only, per [`COLORS.md`](../design-tokens/COLORS.md). For metadata **outside** browse cards (e.g., near forms), keep `text-secondary` — do not use `text-placeholder` next to inputs.

### Every card in a collection shares the same anatomy

All cards within a collection must:

- Use the same slot arrangement
- Display the same categories of information (if one card shows a download count, all cards show it — even when the value is zero)
- Maintain consistent row height in a grid
- Use the same visual treatment for tags, icons, and metadata

Variation comes from content **values**, not from card **structure**.

### Related-item cards on detail pages reuse the browse anatomy

A detail page often shows related items at the bottom. Treat the related-item collection as its own "same collection": same slots, same order as the browse cards for that entity type. Dropping a slot in related-item cards creates an unnecessary second anatomy and forces users to re-learn card reading within the same session.

**If space is constrained,** shrink the card (reduce padding, single-line description, fewer footer stats) **before** dropping a slot.

---

## Layout

Use a responsive grid when content is being browsed, has many items (10+), and items are roughly equal in importance.


Use `<Grid colMinWidth={280}>` to give cards a 280px minimum width and let the browser choose the column count.







Vertically stack cards when each card has substantial content that benefits from full width, the list is short (<10 items), and items follow a natural sequential/chronological order or vary significantly in length.
