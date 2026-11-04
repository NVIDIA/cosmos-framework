# Forms

The anatomy and composition rules for input surfaces that capture user data and submit it to the system. Covers container selection, field anatomy, validation, sizing, and trigger behavior.

**Contents:** [When to use](#when-to-use) · [Anatomy](#anatomy) · [Variants](#variants) · [Composition rules](#composition-rules) · [Modifiers](#modifiers) · [Example application](#example-application)

## When to use

Use a form when capturing structured user input that gets submitted to the system — creating records, editing entities, configuring settings, generating tokens.

## Anatomy

Forms are composed of required parts plus optional parts included when applicable. Wrap each field — input, label, helper text, info, and validation message — in [`FormField`](../components/FormField.md). This file covers form-level composition; `FormField.md` covers per-field mechanics.





### Required parts

| Part | Purpose |
| --- | --- |
| **Field input** | The control that captures the value (TextInput, Select, Checkbox, etc.), wrapped in `FormField` with its label. See [Field input types](#field-input-types) below. |
| **Primary action button** | Submits the form. Label describes the specific action (e.g., "Generate Personal Key"), not generic "Submit". |
| **Cancel / dismiss action** | Always paired with the primary action — close button, Cancel button, or back navigation depending on container. |

### Optional parts

Include when applicable. For per-field help, info, and validation slots, see [`FormField`](../components/FormField.md).

| Part | Include when |
| --- | --- |
| **Error summary** | Form has 7+ fields and submission fails — gives a global view of all errors |
| **Section headings** | Form uses [grouped sections](#sectioning-grouped-fields-on-one-page) |
| **Step indicator** | Form is a [multi-step / wizard](#single-step-vs-multi-step-wizard) |

### Field input types

Most input choices follow from the value's shape: short text, long text, date, and binary values map to the corresponding KUI or native controls. The non-obvious calls:

- **Select vs. RadioGroup** — Select for one choice from 4+ options; RadioGroup for 4 or fewer where seeing all options at once aids the decision.
- **Duration → predefined Select, not a date picker** — when the value is a duration with a fixed option set (e.g. "1 / 3 / 6 / 12 months").
- **[Slider](../components/Slider.md)** — only when the user benefits from seeing their position in a fixed range; always pair it with a numeric input so exact values can be typed.

---

## Variants

### By container

Determine where the form lives — modal, panel, or dedicated page.

| Container | Fields | Choose when |
| --- | --- | --- |
| **Modal (quick action)** | < 7 | ALL true: no conditional fields, no page content to reference, action is reversible, fire-and-forget (user won't revisit submitted values) |
| **Modal (high-stakes)** | 0–3 | ALL true: action is irreversible or high-consequence; user must pause and acknowledge before proceeding |
| **Panel (side panel)** | Any | ANY true: user references page content while filling, split input/output UI, editing an item in a visible collection |
| **Dedicated page** | 7+ | ANY true: complex logic (conditional fields, multi-step config), multiple logical sections, task warrants full attention |

#### When containers overlap

- **< 7 fields but user needs page context** → Panel, not modal. Context visibility overrides field count.
- **< 7 fields but user will revisit submitted values** → Dedicated page or panel. Modals are fire-and-forget — if the user needs to return to review or edit, the form needs a persistent surface.
- **7+ fields but irreversible action** → Dedicated page with a confirmation modal at submit. The page handles complexity; the modal handles the high-stakes acknowledgment.
- **High-stakes but needs page context** → Panel with a confirmation step, not a modal. The panel preserves context; add an inline confirmation or a final modal before the destructive action.

### Single-step vs. multi-step (wizard)

Most forms are single-step — all fields are presented at once within the chosen container (optionally grouped via [sectioning](#sectioning-grouped-fields-on-one-page)).

Use a **multi-step / wizard flow** when:

- The process is complex and long enough that a single page would overwhelm the user
- Later inputs depend on earlier inputs (conditional logic between steps)
- Each step requires explanation or context to complete correctly
- The form primarily serves newer or infrequent users who benefit from guided progression

Wizards add a step indicator to the [Optional parts](#optional-parts) and split the field set across sequential screens, each with its own primary action ("Next") and the final step carrying the true submit.

---

## Composition rules

Apply to all variants.

### Field ordering

Order fields using this priority hierarchy. Evaluate top-down — higher-priority rules override lower ones.

| Priority | Type | Definition | Test |
| --- | --- | --- | --- |
| **1** | **Identity fields** | Value becomes the label or name used to reference this object after creation (seen in lists, tables, navigation) | "Will this field's value be used to identify or reference this object later?" If yes → identity |
| **2** | **Framing fields** | Value exists for human comprehension and does not affect system behavior — captures intent, purpose, or description, not configuration | "Does this field change how the system processes or executes the task?" If no, and it exists so humans understand the purpose → framing |
| **3** | **Configuration fields** | Value directly determines how the system processes or executes the task — changing the value changes the outcome | "Does changing this field's value change what the system does?" If yes → configuration |

#### Within each priority tier — cognitive complexity gradient

Sort fields within each tier from lowest to highest cognitive effort:

1. Fields requiring **free recall** (e.g., typing a name from memory)
2. Fields requiring **selection from a small set of options** (e.g., dropdown with a few clear choices)
3. Fields requiring **domain-specific knowledge** to understand and complete (e.g., selecting training parameters, configuring resource allocations)

#### Override — dependencies

If a field's value determines whether another field appears, changes, or is constrained, that field must be placed before the fields it affects, regardless of its position in the cognitive complexity gradient.

### Trigger behavior

The control that opens the form — usually a button on a page, in a header, or per-row in a table.

- **Placement** — primary page action (fewer than 5 competing actions nearby) → standard button; repeated per row → minimal treatment (icon button or text link).
- **Label** — describe the action **and** the object ("Create Deployment", not "Create" or "New").
    For icon selection, see [`ICONS.md`](../ICONS.md).
  
- **Keep the label even when icon-only is tempting** if the page is dense (icon-only triggers get lost) or the trigger is encountered infrequently (monthly or less — users need the text reminder).

### Validation

> Per-field mechanics (on-blur validation, cross-field-on-submit, focus management) are owned by [`FormField`](../components/FormField.md); message voice and severity by [`error-states.md`](./error-states.md). This section covers only the form-wide layer.

The form-level addition is the error summary as a safety net: on submit with errors, show a banner listing all errors at the top of the form **only for forms with 7+ fields**. For shorter forms, skip the summary and move focus to the first invalid field.

### Button placement and actions

Primary action sits at the bottom, right-aligned or full-width by container (label covered in [Required parts](#required-parts)). For long dedicated-page forms, fix it to the bottom of the viewport so it stays reachable. Always pair it with a cancel / dismiss action:

| Container | Cancel / dismiss treatment |
| --- | --- |
| Modal | Cancel button to the left of the primary action |
| Panel | Close button in the panel header |
| Dedicated page | Cancel button or back navigation |

For destructive submits, use `Button`'s danger variant and require confirmation — see [`confirmations.md`](./confirmations.md).

### Read-only fields

For when to use read-only vs. disabled vs. hidden, see [`element-visibility.md`](./element-visibility.md).

Form-specific:


- Set `readOnly` on the input rather than `disabled` — it stays selectable and keeps the field in the tab order while signaling it can't be edited.






- For read-only key-value pairs in detail-style sections, see [`key-value-display.md`](./key-value-display.md).

---

## Modifiers

### Sectioning (grouped fields on one page)

Within a single-step form, group fields into collapsible/expandable sections with clear headings when:

- The form is long but fields can be logically grouped to support scanning and reduce cognitive load (e.g., contact info, shipping, payment)
- The user benefits from seeing the entire form to review or jump between areas
- The field order is flexible rather than strictly sequential
- Power users need to repeatedly fill out this form quickly (e.g., creating tickets, submitting reports)

Each section should contain related fields that serve a common purpose. Skip sectioning when the form has 7 or fewer fields, all fields relate to a single logical purpose, and no meaningful grouping exists — keep it flat.

### Input width as affordance

Input width communicates the expected length and format of a value before the user types anything. A narrow input signals a short value (zip code, port number); a wide input signals a longer one (full name, URL, description). When width mismatches expectation, users second-guess whether they're entering the right thing.

#### Width tiers

Use consistent width tiers across a form rather than arbitrary pixel values. The tiers below are relative to the form's content column — not the viewport.

| Tier | Width | Use for |
| --- | --- | --- |
| **Extra-small** | 80px | Zip codes, country codes, port numbers, short numeric IDs |
| **Small** | 160px | Phone numbers, dates, currency amounts, short codes |
| **Medium** | 320px | Names, email addresses, single-line labels, hostnames |
| **Large** | 384px | URLs, file paths, longer single-line values |
| **Full** | 100% | Descriptions, notes, queries, any multi-line content (TextArea) |

#### Width rules

| Rule | Rationale |
| --- | --- |
| Width should approximate the longest expected value, not the longest possible value | A full-width input for a 5-digit zip code provides no useful signal. Size for the common case. |
| Select and typeahead width should fit the longest option label without truncation | Truncated options force the user to open the control to read values. Test with real data, not placeholder text. |
| Within a form section, prefer consistent widths when fields share similar value lengths | Mismatched widths between similar fields create visual noise without information value. |
| Do not use width to create visual alignment patterns at the expense of affordance | A form where every field is the same width looks tidy but loses the affordance signal. Affordance accuracy outweighs visual symmetry. |

#### Interaction with form layout

In multi-column form layouts, the column width constrains the input width. When a column is narrower than the ideal tier, the input fills the column width — the column itself becomes the affordance. In single-column layouts, the width tiers above apply directly.

---

## Example application

**Input:** "I need a form for generating an API key in NGC. Fields are key name, expiration, and services included. The user has a list of existing keys visible on the page but doesn't need to reference them while creating a new one."

**Resolved spec:**

1. Container → **Modal (quick action)** (3 fields, no dependencies, no need for page context, low-cost to redo)
2. Flow → **Single-step**, flat (single logical purpose, no grouping)
3. Field order (identity → configuration, then cognitive gradient) →
   - Key Name — identity field (labels the key in lists)
   - Expiration — configuration, low effort (predefined-duration Select)
   - Services Included — configuration, higher effort (Select from services requiring domain knowledge)
4. Trigger → "+ Generate Personal Key" button on the keys list page
5. Widths → Key Name 320px (medium); Expiration 160px (small); Services Included 320px (fits longest service label)
