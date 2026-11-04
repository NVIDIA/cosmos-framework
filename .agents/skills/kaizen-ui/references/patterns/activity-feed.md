# Activity Feed

A chronological list of actor → action → target events, rendered along a vertical timeline spine.

## When to use

Use an activity feed when:

- A history of actions, events, or changes needs to be visible to the user
- The content is chronological — newer events are more relevant than older
- Users need to monitor, review, or act on events that have occurred
- Multiple actors or systems generate activity over time

Don't use when:

- The data is better represented as a table (structured logs with many sortable/filterable columns). See [data-display.md](./data-display.md) — pick a table when the primary task is sorting/filtering across many columns; pick this feed when the primary task is reading a narrative history. (data-display also lists "audit logs" as a candidate for a collapsible section — that's only for _secondary, tucked-away_ history; a primary activity/history surface uses this pattern.)
- The content is not chronological
- There's a single actor and no collaboration or system events to track

**Placement on a detail/workspace page:** history and activity outgrow a side panel. Put the feed on a full page or History/Activity tab. See [detail-placement.md](../layouts/detail-placement.md). Timelines flow sequentially, so avoid an extra visual container around the entire feed.

## Anatomy

Each entry is composed of required parts plus optional parts included when applicable.

### Required

| Part | Purpose | Notes |
| --- | --- | --- |
| **Actor marker** | Who or what performed the action | [`Avatar`](../components/Avatar.md) + name for users; [StatusIndicator](../components/StatusIndicator.md) for system events. See [Composition rules → Status indicator](#status-indicator-dot-not-icon). |
| **Action** | What happened | Natural language: "left a comment in", "completed backup", "training failed" |
| **Meta line** | When + where | Single text-secondary line: `timestamp · source` (e.g., "Apr 28, 2026, 8:30 AM · System"). Relative time for events < 24hrs ago i.e. ("2 hours ago"), absolute timestamp otherwise |
| **Target link** | Navigable context | What the action was performed on, rendered as an [`Anchor`](../components/Anchor.md) to the relevant item |

### Optional

Include when applicable.

| Part | Include when |
| --- | --- |
| **Detail body** | 1–2 lines explaining implication or auto-action ("Automation opened ticket NV-44102; rack isolated.") |
| **Detail metrics** | Quantitative details immediately useful without clicking in ("Size: 23 GB", "Duration: 4m 32s") |
| **Progress indicator** | Event relates to a measurable state with a threshold or capacity (memory at 87%) |
| **Error message** | Event is a failure and the user needs the cause |
| **Tags** | Event needs categorization or identifiers cross-reference other systems |
| **Inline actions** | User commonly needs to respond immediately ("Retry", "View Details", "Dismiss") |

---

## Variants

| Variant | Marker | Use when |
| --- | --- | --- |
| **User activity** | Avatar + name | Tracks actions performed by people |
| **System event** | Colored **status dot** | Tracks automated processes, system health, or operational events |
| **Mixed** | Both, differentiated visually | Both user and system entries belong in the same timeline |

For mixed feeds: keep entry structure consistent across types and sort chronologically regardless of type.

---

## Composition rules

Apply to all variants.

### Status indicator: dot, not icon

System event entries use a medium labeled status dot, color-coded by severity. Do not add a checkmark, triangle, X, or another shape glyph.

| Severity | Color |
| --- | --- |
| Error | red |
| Warning | yellow |
| Success | green |
| Info | blue |

**Why a dot, not an icon?** A dot is the lightest possible status signal — it communicates severity through color alone, leaving the entry's text content to do the explanatory work. Icons add a second visual element (shape) that competes with the title for the eye, makes dense feeds feel cluttered, and forces a second decoding step.

The dot needs an `aria-label` describing the severity for screen-reader users (the color carries the meaning visually).

### Timeline spine is required

A visible 1px vertical line on the left side of the entry list, connecting markers. Without it, the pattern reads as a plain list rather than a timeline.

- The status marker sits centered on the spine, over the spine rather than under it.
- The spine is **suppressed on the last entry** so the timeline visually closes.
- The marker carries severity via color + `aria-label`; the spine itself is `aria-hidden`.

**Anti-pattern:** rendering each entry as a standalone `<li>` with a leading dot and no connecting line. Without the spine, the visual chain between events is broken and the feed reads as a flat list.

### System event entry: 3-line composition

| Line | Content | Style |
| --- | --- | --- |
| 1 | Status dot + event title | Title is `body/bold/md`. Dot sits at the same baseline as the title's first character. |
| 2 | Timestamp · Source (e.g., "Apr 28, 2026, 8:30 AM · System") | `body/regular/sm` + secondary color. The `·` separator distinguishes the two metadata items without using a comma (which reads as part of the timestamp). |
| 3 | Description body | `body/regular/sm` or `body/regular/md`. Optional but typical — explains the event's implication or auto-action. |

User-actor entries follow the same layout with the avatar replacing the dot on line 1.

---

## Modifiers

### Time grouping

Group entries by time period when the feed spans multiple days or users scan for a specific period.

| Grouping | Use when |
| --- | --- |
| Today / Yesterday / [Date] | Daily activity matters |
| This Week / Last Week / [Week of Date] | Less frequent activity |
| This Month / [Month Year] | Long-running historical feeds |

Place the group heading above the first entry in that group, with clear visual hierarchy distinguishing it from entries. Skip grouping for recent-only feeds (last few hours) — chronological order with timestamps is enough.

### Sizing

| Size | Use when |
| --- | --- |
| **Default** | Single-column reverse-chronological timeline, full container width |
| **Compact** | Sidebar/panel with limited width, high volume activity, or only essential info (actor, action, time) is needed. Omit metrics, progress, and inline actions — surface on click/expand. |
| **Expanded / detail** | Feed is primary page content, events carry significant detail (metrics, errors, progress) users need without clicking, activity volume is lower |

### Loading and pagination

- Load most recent entries first (newest at top)
- Use infinite scroll or a "Load more" button for older entries rather than numbered pagination
- For live updates: show a "New activity" indicator at the top rather than auto-pushing content down (auto-push disorients the user's reading position)
