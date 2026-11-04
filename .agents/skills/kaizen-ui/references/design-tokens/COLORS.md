# Semantic color tokens

Theme-aware color tokens for backgrounds, foregrounds, and borders across KUI implementations. **Default to these over primitives** — they adapt to light/dark themes and encode intent.

## Roles

Tokens are organized by **role** — pick the section that matches what you're styling.

| Role                 | When to use                                                           |
| -------------------- | --------------------------------------------------------------------- |
| Surface & Background | Container/page chrome backgrounds.                                    |
| Foreground           | Text and icon colors.                                                 |
| Feedback             | Status messaging — info, warning, success, danger.                    |
| Severity             | Ranked issues, alerts, and incidents on an established `0`–`5` scale. |
| Data visualization   | Quantitative displays, selected by the relationship between values.   |
| Interaction          | Interactive controls and their states.                                |
| Accent               | Non-semantic color tagging (chips, badges, color-coded categories).   |
| Component            | Component-specific tokens (tooltips, skeletons, scroll tracks).       |

Many tokens append a **state suffix**:

| State            | Meaning                                           |
| ---------------- | ------------------------------------------------- |
| (none) / `-base` | Default, inactive state.                          |
| `-hover`         | User is positioned over the element.              |
| `-pressed`       | User has clicked or tapped.                       |
| `-selected`      | Element is highlighted as selected.               |
| `-disabled`      | Element is interactive but currently unavailable. |
| `-subtle`        | Lower-prominence variant of the same role.        |
| `-strong`        | Higher-prominence variant of the same role.       |

> Token-name tables below list Tailwind classes. The corresponding CSS variable is the same name with the property prefix restored (`bg-surface-raised` ↔ `var(--background-color-surface-raised)`)

## Surface and background

**Surface** tokens stack to create hierarchy (sunken → base → raised → overlay). In light mode, pair surfaces with shadows since adjacent surfaces share a color.

| Class                   | When to use                                                                             |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `bg-surface-sunken`     | Lowest surface — backdrop for content. Pair with `bg-surface-base` to create hierarchy. |
| `bg-surface-base`       | Default background for sections and contained elements.                                 |
| `bg-surface-raised`     | Cards or elements that sit above sunken/base — adds emphasis.                           |
| `bg-surface-overlay`    | Highest surface — menus, popovers, anything floating over page content.                 |
| `bg-surface-navigation` | App navigation chrome (sidebars, top bars).                                             |

**Background** tokens are similar but don't require shadows — use them for neutral separation between page regions.

| Class                     | When to use                                                              |
| ------------------------- | ------------------------------------------------------------------------ |
| `bg-background-default`   | Neutral page background.                                                 |
| `bg-background-subtle`    | Slightly de-emphasized region — alternate row stripes, secondary panels. |
| `bg-background-emphasis`  | Slightly emphasized region — highlight a section without raising it.     |
| `bg-background-highlight` | Neutral emphasis for callouts or emphasized panels.                      |
| `bg-background-contrast`  | High-contrast region against the default surface.                        |

## Foreground

Use foreground tokens for text and icons. Start here before reaching for more specialized roles.

| Class                | When to use                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `text-primary`       | Key content — primary information or actions.                          |
| `text-secondary`     | Secondary information, such as labels.                                 |
| `text-disabled`      | Disabled text and icons.                                               |
| `text-placeholder`   | Hints inside empty inputs and helper text.                             |
| `text-inverse`       | Text on inverted/contrast backgrounds (e.g. `bg-interaction-inverse`). |
| `text-inverse-brand` | Brand-green accents on inverted backgrounds.                           |
| `text-base`          | Generic body color - meant for icons.                                  |

## Feedback

Feedback tokens carry semantic meaning — reserve them for status messaging, alerts, and validation. Each kind ships with a matched `bg-/text-/border-` triplet at the base level; `danger` is the only kind with extra states.

| Kind    | When to use                                                                                                                                        |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Info    | Supplementary information, neutral guidance, low-risk status messaging. Triplet: `bg-feedback-info`, `text-feedback-info`, `border-feedback-info`. |
| Warning | Risks, caution states, guidance to help users avoid mistakes. Triplet: `bg-feedback-warning`, `text-feedback-warning`, `border-feedback-warning`.  |
| Success | Positive outcomes, confirmations, completed states. Triplet: `bg-feedback-success`, `text-feedback-success`, `border-feedback-success`.            |
| Danger  | Errors, destructive actions, critical attention states. Triplet plus the danger-only states below.                                                 |

Each kind also exposes a `text-feedback-{kind}-inverse` for use on saturated/feedback-colored backgrounds.

### Danger-only states

| Class                               | When to use                                                    |
| ----------------------------------- | -------------------------------------------------------------- |
| `bg-feedback-danger-hover`          | Hover over a danger-colored interactive element.               |
| `bg-feedback-danger-pressed`        | Pressed/active danger element.                                 |
| `bg-feedback-danger-strong`         | High-contrast danger background (e.g. saturated alert banner). |
| `bg-feedback-danger-subtle-hover`   | Hover over a subtle danger surface.                            |
| `bg-feedback-danger-subtle-pressed` | Pressed subtle danger surface.                                 |
| `text-feedback-danger-strong`       | Higher-emphasis danger text (e.g. error headlines).            |
| `text-feedback-danger-subtle`       | De-emphasized danger text (e.g. inline validation hints).      |
| `border-feedback-danger-hover`      | Hover state on a danger-bordered control (e.g. invalid input). |
| `border-feedback-danger-strong`     | Higher-emphasis danger border.                                 |
| `border-feedback-danger-subtle`     | De-emphasized danger border.                                   |

## Severity

Use severity tokens only when the experience has an established `0`–`5` severity model. Lower numbers indicate higher severity; use `severity-none` when no severity is assigned. Do not substitute this scale for feedback tokens in general status, validation, or action messaging.

Use the CSS variable `var(--color-severity-{level})`, replacing `{level}` with `0`, `1`, `2`, `3`, `4`, `5`, or `none`. With the Tailwind plugin, use a property-specific utility such as `bg-severity-0`, `text-severity-0`, or `border-severity-0`.

Always pair severity color with a visible label, icon, or other indicator so the meaning does not depend on color alone.

## Data visualization

Charts, graphs, meters, and legends have their own token families, selected by the relationship between the values rather than by role. They also break the variable-naming rule above, since they sit in the `--color-*` namespace. Read [CHART_DATA_COLORS.md](./CHART_DATA_COLORS.md) before coloring a quantitative display.

## Interaction

Use interaction tokens for interactive controls (buttons, inputs, toggles) and the borders that communicate affordance. Generic borders (`border-base`, `border-brand`, `border-disabled`) live here too.

The shape mirrors the state suffix table above. Three families:

- `*-interaction-{base,hover,pressed,selected}` — default neutral interactive treatment.
- `*-interaction-primary-{base,hover,selected}` — brand-green primary action treatment.
- `*-interaction-inverse-{base,hover,pressed}` — dark-on-light treatment for high-contrast actions.

Plus disabled coverage: `bg-interaction-disabled`, `bg-interaction-disabled-checked`, `text-interaction-disabled-checked`, `text-interaction-selected`, and `border-interaction-{base,hover,pressed,selected,strong,disabled,inverse,inverse-hover,inverse-pressed,primary-base,primary-hover,primary-selected}`.

Generic borders not tied to interactive state: `border-base`, `border-brand`, `border-disabled`.

## Accent

Accent tokens are **non-semantic** color tags — useful when our semantic tokens don't fit. Common cases: chips, badges, color-coded categories, status pills that don't map to feedback states.

The palette is a clean matrix:

- **Colors:** `blue`, `gray`, `green`, `purple`, `red`, `teal`, `yellow`. (`black` and `white` exist but only for `text-` and `border-` — no backgrounds, no states.)
- **Properties:** `bg-accent-{color}`, `text-accent-{color}`, `border-accent-{color}`.
- **Variants per color:** `-subtle` (light tint, suitable for chip fills), no suffix (mid-tone), `-strong` (saturated for emphasis).
- **States per variant:** `-hover`, `-selected` (e.g. `bg-accent-blue-hover`, `bg-accent-blue-subtle-hover`).

So a chip with a blue tint that highlights on hover is `bg-accent-blue-subtle` + `bg-accent-blue-subtle-hover`. A saturated red badge is `bg-accent-red-strong`.

## Component

Component-specific tokens. Reach for these only when styling the matching component pattern.

| Token                          | When to use                                               |
| ------------------------------ | --------------------------------------------------------- |
| `bg-component-tooltip`         | Tooltip background. Pair with `border-component-tooltip`. |
| `border-component-tooltip`     | Tooltip border.                                           |
| `bg-component-skeleton`        | Skeleton loading placeholder default fill.                |
| `bg-component-skeleton-subtle` | Lower-contrast skeleton variant.                          |
| `bg-component-track`           | Scroll track and slider track background.                 |
| `bg-component-track-inverse`   | Track background on inverted/contrast surfaces.           |
