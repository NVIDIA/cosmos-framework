# Settings

A row-based pattern for user-configurable preferences.

## When to use

Use the settings pattern when:

- Users need to configure preferences, behavior, or state that persists across sessions
- Each option maps to a single discrete control (switch, select, input, slider)
- The configuration is ongoing — not a one-time setup wizard or data-entry form

For traditional create / edit data-entry forms, see [`forms.md`](./forms.md).

---

## Anatomy

Every setting in a settings surface shares the same row structure. Variation comes from the control type and value, not from which parts are present.

### Required

| Part | Purpose |
| --- | --- |
| **Title** | Names the setting. Clear and concise — the user should understand what this controls from the title alone. |
| **Control** | The interactive element that changes the setting value. See [Composition rules → Choose the right control](#choose-the-right-control). |

### Optional

| Part | Include when |
| --- | --- |
| **Description** | Title alone is ambiguous, or the setting's impact is not immediately obvious. Brief explanation of what the setting does or what its effect is. |
| **Section heading** | Settings split across distinct categories. See [Composition rules → Grouping](#grouping). |
| **Save bar** | Surface uses explicit save (vs. immediate-apply). See [Composition rules → Save behavior](#save-behavior). |

### Row layout

Each setting occupies a single horizontal row with the title and description on the left and the control on the right. Settings stack vertically within their section.

Use [FormField](../components/FormField.md) to keep each control's label, description, and validation relationships together. Lay out each row with the framework's Stack/Flex primitives or equivalent CSS, and use [Divider](../components/Divider.md) only when spacing does not separate rows clearly enough.

---

## Variants

| Variant | Use when |
| --- | --- |
| **Dedicated settings page** | Settings are application-level (not contextual to a specific view), there are many settings across multiple categories, or the user is performing a focused configuration task rather than a quick tweak. |
| **Settings panel (side panel / drawer)** | Settings are contextual to the current view or tool, the user needs to adjust settings while seeing the effect on the main content, or the settings set is small (fewer than 10 settings). |

---

## Composition rules

Apply to all variants.

### Choose the right control

Match the control to the shape of the setting's value. The table below covers the most common settings controls — treat them as recommended defaults for consistency, not an exhaustive list. Any KUI form control is available.

| Control | Recommended when |
| --- | --- |
| **Toggle / Switch** | Binary on/off state that applies immediately. The most common settings control — communicates immediacy. |
| **Select** | More than 2 predefined options; one value from a fixed set. Examples: language, timezone, notification frequency, default view mode. |
| **Text input** | Setting value is user-defined and freeform. Examples: display name, custom domain, webhook URL. |
| **[Slider](../components/Slider.md)** | Setting value is numeric within a defined range and the user benefits from seeing their position within the range. Examples: volume, notification threshold, data retention period. |

### Grouping

| Organization | Use when |
| --- | --- |
| **Sections with headings** | Settings page contains more than 5 settings, or settings serve distinct categories (e.g., "Account", "Notifications", "Privacy", "Appearance"). Each section heading describes the category. Separate sections with visible dividers or generous spacing. |
| **Flat list (no headings)** | Settings page contains 5 or fewer settings, and all settings relate to a single category or purpose. |

### Save behavior

| Model | Use when |
| --- | --- |
| **Immediate application** (default for toggles) | Changes take effect as soon as the user interacts with the control. No save button. Use for toggles and simple single-value selections. |
| **Explicit save** | Multiple settings may need to be changed together before taking effect; changes have significant consequences the user should commit to deliberately; or the surface contains text inputs where the user needs to finish typing before applying. |

When using explicit save, include a cancel / discard option and indicate unsaved changes visually.

#### Save bar

The explicit-save affordance is a persistent bar holding the commit actions. Anatomy and rules:

- **Actions:** `[Cancel] [Save]`, with Save as the trailing brand-primary action. Disable both until there are unsaved changes; while saving, show a busy state on Save and keep Cancel available.
- **Placement:** bottom-aligned within the settings surface. On a dedicated page, a sticky footer bar; in a settings panel, pinned to the panel footer. Keep it visible without scrolling away from long forms.
- **Unsaved-changes signal:** indicate the dirty state explicitly (e.g. an "Unsaved changes" label near the bar) and warn before navigation away discards edits.
- **Result feedback:** on success the in-place control values are the confirmation — add a `Toast status="success"` only when the result scrolls out of view. On failure follow [error-states.md](./error-states.md) (inline for field-level rejections; transient toast/notification for the save action itself). See [feedback.md](./feedback.md) for the surface decision.

---

## Example application

**Input:** "I need a settings panel for a data visualization tool. Users can set default chart type (bar, line, pie), and configure auto-refresh interval."

**Resolved spec:**

1. Variant → **Settings panel** (contextual to the tool, user wants to see the effect, 2 settings)
2. Grouping → **Flat list** (2 settings, single category)
3. Controls →
   - Default chart type: **Select** (3+ predefined options, single selection)
   - Auto-refresh interval: **Select** or a **Slider** depending on whether values are predefined steps or a continuous range
4. Save behavior → **explicit save** (with a [save bar](#save-bar)) for the chart type and interval if changes affect saved preferences.
