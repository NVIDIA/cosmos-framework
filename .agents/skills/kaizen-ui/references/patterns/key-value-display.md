# Key-Value Display

Labeled read-only value pairs

## When to use

Use [KeyValueGroup](../components/KeyValueGroup.md) for labeled read-only values anywhere in the UI. For **editable** fields, use form fields. For tabular data with many rows, see [data-display](./data-display.md).

In a [SidePanel](../components/SidePanel.md) (the common detail-metadata surface), render `KeyValueGroup` directly in the panel body rather than nesting another container. See [detail-placement.md](../layouts/detail-placement.md) for what metadata belongs in a panel vs. a full page.

## Anatomy

`KeyValueGroup` renders a native description list (`<dl>`) where each item wraps a muted `<dt>` key and a primary `<dd>` value.

| Part      | Typography         | Color            | Notes                        |
| --------- | ------------------ | ---------------- | ---------------------------- |
| **Key**   | `label/regular/sm` | `text-secondary` | `<dt>`, short noun phrase    |
| **Value** | `body/semibold/md` | `text-primary`   | `<dd>`, text or rich element |

---

## Orientations

Two axes control layout:

- **Group `orientation`** (`"vertical"` default | `"horizontal"`): whether the pairs themselves stack in a list or sit side by side in a wrapping row.
- **Pair `pairOrientation`** (`"horizontal"` default | `"vertical"`): whether each key sits beside its value or stacks above it.

| Combination | Layout | Use when |
| --- | --- | --- |
| Vertical group + horizontal pairs (**default**) | Stacked rows; all keys share one aligned column (at least `100px` wide, up to `40%` of the group) | Default for detail panels, cards, and settings summaries with concise keys and values. |
| Vertical group + vertical pairs | Stacked rows; each key sits above its value (`4px` gap) | Narrow columns (such as a sidebar) or groups where values are long sentences/URLs. |
| Horizontal group + horizontal pairs | Wrapping inline row of `key value` pairs (`40px` between pairs, `8px` between key and value) | Compact summary strips above a table or in a card footer. |
| Horizontal group + vertical pairs | Wrapping row of stacked pairs (`16px` between pairs) | Metadata bars under a page header. |

One pair in an aligned list can opt out to vertical when a single field holds long-form prose such as a description or notes.

---

## Composition rules

### Rich value types

The value is not always plain text. Match the value content to the data; do not wrap plain text in `Text` because `<dd>` already applies `body/semibold/md`:

| Value type | Render as | Notes |
| --- | --- | --- |
| **Plain scalar** (version, date, GPU type) | Plain text inside `<dd>` | Default. Render missing values as `—` with visually hidden `"Not provided"` text, never an empty `<dd>`. |
| **Status / state** (running, failed) | [`Badge`](../components/Badge.md) | Semantic color with text, never color alone. |
| **Owner / person** | [`Avatar`](../components/Avatar.md) marked `aria-hidden` + name text, both inside the value | The name carries the meaning, so the avatar is not announced. |
| **Leading icon** (timestamp, location) | Icon + text, both inside the value | Icons hide themselves from assistive technology; the text carries the meaning. |
| **Tags / categories** | Non-interactive [`Tag`](../components/Tag.md) | Use a neutral gray outline treatment, not Badge. Reserve Badge for status. |
| **Link / external resource** | [`Anchor`](../components/Anchor.md) | Keep at most one interactive control per value. |

A value should never become an editable form control — if it does, the surface is a form, not a key-value display.

### Default stacked group (aligned key column)


```tsx
<KeyValueGroup
  items={[
    { slotLabel: "Owner", slotValue: "Jordan Lee" },
    {
      slotLabel: "Status",
      slotValue: <Badge color="green">Active</Badge>,
    },
    { slotLabel: "Region", slotValue: "us-west-2" },
  ]}
/>
```







### Stacked pairs (label above value)


```tsx
<KeyValueGroup
  pairOrientation="vertical"
  items={[
    { slotLabel: "Owner", slotValue: "Jordan Lee" },
    { slotLabel: "Region", slotValue: "us-west-2" },
  ]}
/>
```







---

## Modifiers

### Grouping within and between sections

Give each `<dl>` a visible section heading (`<h2>` / `<h3>`). A `<dl>` has no accessible name of its own, and screen readers expose `aria-label` on it inconsistently.

| Scope                  | Spacing                                   |
| ---------------------- | ----------------------------------------- |
| Pairs within a section | Handled by `KeyValueGroup` (`16px` / `4`) |
| Between sections       | Sectional spacing (`24px`) + a heading    |


```tsx
<Stack gap="4">
  <Text kind="title/sm" asChild>
    <h2>Workload</h2>
  </Text>
  <KeyValueGroup
    items={[
      { slotLabel: "Owner", slotValue: "Jordan Lee" },
      { slotLabel: "Region", slotValue: "us-west-2" },
    ]}
  />
</Stack>
```







### Horizontal row of pairs

For metadata bars under a page header or summary strips above a table, set `orientation="horizontal"` (and `pairOrientation="vertical"` when each label should stack above its value).


```tsx
<KeyValueGroup
  orientation="horizontal"
  pairOrientation="vertical"
  items={[
    { slotLabel: "Projects", slotValue: "8" },
    { slotLabel: "Storage", slotValue: "640 GB" },
    { slotLabel: "Plan", slotValue: "Team" },
  ]}
/>
```






