# Icons

Decision rules for KUI iconography: whether to use an icon, which variant, and which icons are safe icon-only.

## Setup

We use [NVIDIA Icons](https://github.com/NVIDIA/icons#documentation) - specifically the React package. Refer to that packages README via the link or node_modules for setup documentation if you're configuring it for the first time or debugging.

Icons come from `@nvidia/react-gui-icons/icons` as named PascalCase exports.

Names that would start with a digit or shadow a JS global are prefixed `Icon` — `IconError`, `IconImage`, `IconDocument`, etc.

```tsx
import { ChevronDown } from "@nvidia/react-gui-icons/icons";

<ChevronDown />                  // line/outline (default)
<ChevronDown variant="fill" />   // solid; prefer at smaller sizes
```

Icons inherit `currentColor`, so set the color on an ancestor rather than on the icon. `size` is a CSS length string (`size="24px"`), defaulting to `1em`.

## Decision 1: Use an icon at all?

Icons are not free — they add visual density, scanning cost, and inconsistency risk if applied unevenly.

| Pattern | Use when | Examples |
|---|---|---|
| **Label only** | The action is a verb-clear contextual choice. The label already communicates intent; an icon would be decorative or ambiguous. Default for review actions, form submits, and most secondary buttons. | Approve, Decline, Save, Cancel, Edit, Submit, Delete (use color, not icon, to signal danger) |
| **Icon + label** | The action belongs to a recognizable system class (navigation, search, refresh, add, export, sort, filter, upload). The icon clarifies the action's _category_, not its _target_. | Refresh, + Add cluster, Search, Export, Filter, Sort |
| **Icon only** | Tight UI surfaces where the icon is universally understood. | Close, kebab, expand/collapse, table-row delete in dense rows |

**Default to "label only" when uncertain.** Ask whether the icon is doing real work, not decorating. Adding icons indiscriminately makes UI feel busy without improving comprehension; removing icons that genuinely aid recognition (refresh, search, add) makes scanning slower.

---

## Decision 2: Which icon?

Cheatsheet of most common scenarios:

| Action | Icon |
|---|---|
| Create / Add | `Add` (plus sign) |
| Edit | `Pencil` |
| Delete / Remove | `Trash`/`TrashDelete` |
| Save / Download | `Download` |
| Copy / Duplicate | `CopyGeneric`/`CopyDoc`(for document specifically) |
| Close / Dismiss | `Close` |
| Search | `MagnifyingGlass` |
| External link | `OpenExternal` |
| Filter | `Filter` |
| Sort | `Sort`(no arrow), `SortAscending`, `SortDescending`|
| Refresh / Reload | `Refresh` |
| Settings | `Cog` |
| More actions (kebab) | `MoreVert` |
| More actions (horizontal) | `MoreHoriz` |

If none of those fit your use case see [`../assets/icon-catalog.md`](../assets/icon-catalog.md) for the full catalog grouped.

**Toolbars and PageHeader actions:** when a button needs an icon render it leading

---

## Decision 3: Solid vs. outline variant

Icons accept a `variant` prop defaulting to `variant="line"` (outline). `variant="fill"` is solid.

Use solid icons for navigation, small icon sizes for improved readability, or primary/prominent actions - this provides stronger visual weight and easier to recognize at smaller sizes.

---

## Icon-only safe set

These icons are universally recognized enough to omit a visible text label. **All icon-only usage must be coupled with a11y considerations such as a `aria-label` on the wrapping element.**

| Icon | Action |
|---|---|
| `Close` | Dismiss / close |
| `Pencil` | Edit |
| `Trash` | Delete |
| `Cog` | Settings |
| `MoreVert` / `MoreHoriz` | Overflow menu |
| `MagnifyingGlass` | Search |
| `ChevronDown` / `ChevronUp` | Expand / collapse |
| `ChevronLeft` / `ChevronRight` | Previous / next |
| `Add` | Create / add |

Every other icon should be paired with a text label.

---

## Gotchas

- **Consistency within a group is non-negotiable.** Within a single toolbar, button group, badge row, or menu, every item either has an icon or none does. Mixing is the most common iconography mistake and reads as visual sloppiness.
- **Don't mix solid and outline within the same visual group** (toolbar, menu, tab set, nav). Mixed variants imply a semantic difference where none exists.
- **Icon-only buttons must set `aria-label`.** Screen readers otherwise announce "button" with no context.
- **Don't force an icon when no recognizable metaphor exists.** A wrong metaphor teaches users an incorrect mental model — worse than no icon. Use a text-only label.
- **Don't add an icon to every badge, tag, or table cell.** Icons in every cell become visual noise and lose the emphasis that selective use provides.
- **Prefer plain `Check` over `CheckCircle` in dense contexts** (table cells, small badges). The enclosing circle adds visual weight and reduces readability at small sizes.
