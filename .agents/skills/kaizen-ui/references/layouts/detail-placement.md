# Detail placement

Side panel, standalone page, or both.

A decision guide for where a single entity's detail lives in the IA. Answer two questions, read off the outcome, then follow only the section your outcome points to.

---

## Step 1 — Does the detail outgrow a panel?

**Yes** if _any_ of these are true:

- Entity has **nested sub-resources** (child records, related entities, logs, events)
- Entity accumulates **history or activity** over time (status changes, comments, runs)
- Detail requires **complex inline actions** (multi-step forms, threaded comments, long-running workflows)
- Content needs **tabs** (e.g. Details / Logs / History)
- A **narrow viewport (< 720px)** is a primary target

Otherwise the content is **flat** (no sub-resources, no history, no tabs) and fits a panel.

## Step 2 — Is the entity consumed from a list?

**Yes** if _any_ of these are true:

- User reviews items **sequentially** from a list (triage, queue processing)
- User needs **list context while reviewing** (filters, position in queue, comparison across records)
- **Different personas** use the same entity at different depths

Otherwise the entity is reached directly (search, deep link, single record).

## Outcome

|                          | From a list — **No**     | From a list — **Yes**             |
| ------------------------ | ------------------------ | --------------------------------- |
| **Outgrows panel — No**  | Standalone page (simple) | **Side panel only**               |
| **Outgrows panel — Yes** | **Standalone page**      | **Both** (progressive disclosure) |

If the signals are mixed or ambiguous, ask the user. Do not silently choose — the decision has lasting information-architecture consequences.

> The sections below apply **only if your outcome includes a side panel** (Side panel only or Both). If your outcome is a standalone page, you're done.

---

## Progressive disclosure (panel → page)

Applies to the **Both** outcome.

1. **List row** — scannable summary: name, status, key attribute, timestamp.
2. **Side panel** — glanceable detail not in the list: full metadata, status context, operator actions. Opens as a right overlay without leaving the list; the list stays primary.
3. **Full page** — everything: complete history, activity thread, related entities, complex actions, tabs.

When both exist, the side panel must include a **"View full details"** link to the standalone page.

## What goes in the panel vs. the page

| Content                                      | Side panel         | Full page            |
| -------------------------------------------- | ------------------ | -------------------- |
| Identity (name, ID, type)                    | ✓                  | ✓                    |
| Status + key timestamps                      | ✓                  | ✓ (mini-tile grid)   |
| Core metadata (team, requester, capacity)    | ✓                  | ✓                    |
| Latest operator note or message              | ✓ (most recent)    | ✓ (full thread)      |
| Status history / activity timeline           | ✗                  | ✓                    |
| Actions (approve, decline, reply)            | Quick actions only | Full action set      |
| Related entities (linked cluster, inventory) | Link only          | Inline or expandable |
| Collapsible configuration sections           | ✗                  | ✓                    |
| Tabs (Details, Logs, History)                | ✗                  | ✓                    |

**Rule of thumb:** "what is this and what's its current state?" → panel. "what happened, why, and what should I do about it?" → full page.
