# Modifying an existing page

When adding new content to an existing page, the question is not _whether_ to add it — that's a product decision — but _how_: append in place, restructure into tabs/sections, or split into a sub-page.

## What counts as "substantial"

Re-evaluate structure if **any** of these are true:

- 3 or more new fields, columns, or items
- Any new section heading
- Any new sub-resource (a child entity with its own state, list, or lifecycle)
- Any change that would push primary content below the fold at 1280px

A single field, a single row, a label rename, or a copy-only change does not trigger restructuring. Append in place.

## Decision: append, restructure, or split?

| Signal | Append in place | Restructure into tabs / sections | Split into sub-page |
| --- | --- | --- | --- |
| **Fits existing IA**  | Extension of what the page already does | Introduces a new dimension (activity, sub-resources, settings) | A different "thing" with its own state and lifecycle |
| **User mental model** | Same task | Related tasks reachable from one entity| Distinct task warranting its own URL |
| **Content density**   | Page still scans within typical viewport| Page would require scrolling past unrelated content| Stuffing it in would push primary content below the fold |
| **Discoverability**   | Visible alongside existing content | Tab labels make the addition findable  | Sub-page is reachable via clear navigation (breadcrumbs, sub-nav, links from parent) |
| **Coherence with siblings** | Sibling pages look similar after the change | Tabs add a consistent structure across siblings | Sub-page matches the pattern other entities use (e.g., every cluster has a Nodes sub-page) |

**When the signals are mixed, prefer restructuring (tabs) over splitting (sub-page).** Tabs preserve discoverability and keep the user anchored to the entity. Splitting is heavier — only justified when the new thing is independently navigable.

## Anti-patterns

- **Appending until the page no longer scans.** If a designer has appended 4+ times across iterations, restructure now.
- **Tabs as a dumping ground.** A tab should represent a coherent dimension (Activity, Settings, Configuration). No "Misc" or "More" tab.
- **Sub-page unreachable by UI.** If you split into a sub-page, ensure breadcrumbs, sub-nav, or in-page links make it discoverable.
- **Diverging structures across sibling entities.** If clusters have tabs but networks don't, sibling pages drift.
