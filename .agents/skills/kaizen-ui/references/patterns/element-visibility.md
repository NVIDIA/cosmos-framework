# Element Visibility & Interactivity

Decides whether an element should be **hidden**, **read-only**, or **disabled** instead of enabled. This is a judgment about informational value and path-to-action — component files cover the "how." Elements earn visibility by providing value, and earn interactivity by having a clear, achievable path to enablement.

## When to apply

Apply to every interactive or labeled element you are about to render. The output is one of: hidden, read-only, disabled, or the default enabled state.

## Decision framework

Ask in order; stop at the first decisive answer.

| # | Question | If NO | If YES |
| --- | --- | --- | --- |
| 1 | Does the user benefit from seeing this element? | **Hidden** | Continue to Q2 |
| 2 | Is it informational rather than interactive? | Continue to Q3 | **Read-only** |
| 3 | Can the user act on it right now? | Proximate path to enable? Yes → **Disabled**, No → **Hidden** | **Enabled** |

The non-obvious calls live in Q2 and Q3: read-only vs. disabled, and disabled vs. hidden. The rest is straightforward — hide what the user can't access, can't benefit from, or isn't authorized to see.

## Read-only vs. disabled

Different states with different semantics; agents routinely conflate them.

- **Read-only** = a value the user should see but cannot change. It is _not a control_. Use `readonly` / `aria-readonly`, or a display element such as `<dd>` — **never `disabled`**. Typical cases: system-generated values (IDs, timestamps, computed fields), values inherited/locked from a parent, a prior workflow step shown for reference, or context for an adjacent editable field.
- **Hidden** = gone for everyone: conditional rendering (preferred), the `hidden` attribute, or `display: none` — each already drops the element from the accessibility tree, so adding `aria-hidden` on top is redundant. `visibility: hidden` hides it from assistive tech too but keeps its layout space, leaving a gap. Reserve `aria-hidden="true"` for content that stays on screen, such as a decorative glyph beside its own text label, and never put it on anything focusable.
- **Disabled** = a control that _could_ be interactive but isn't right now — and only when there's a proximate path to enable it (below).

## Disabled requires a proximate path

Disable only when the user can discover and complete the unlock action without leaving the current task — the prerequisite is visible on the page or in the current flow.

- Submit disabled until required fields are filled (prerequisites on the same form)
- Deploy disabled until config validates (status shown on the same page)

**No proximate path → hide instead.** A disabled control with no discoverable unlock is interface noise — it provokes "why can't I click this?" without telling the user what to do.

### Disabled must explain itself

Pair every disabled control with an _actionable_ reason — state the prerequisite, not the restriction.

- **Tooltip or accessible description** (default for buttons, actions, standalone controls): "Select a source dataset to enable export." Not "This action is unavailable."
- **Inline `FormField` helper text** when the dependency is a visible parent→child field relationship: "Available after selecting a region above."
- Keep it to one short sentence. If the explanation needs more, the path probably isn't proximate — hide instead.

Gotcha: disabled elements can't receive focus, so a tooltip trigger may need to wrap the element (or use `aria-describedby`) to stay keyboard-accessible.

## Exceptions

### Spatial consistency

In high-frequency control surfaces (toolbars, action bars used many times per session) where users build muscle memory, you may show an element as disabled even without a proximate path, purely to keep the layout stable. All must hold:

1. The surface is used at high frequency with a stable, cross-session layout
2. Removing the element would reflow or shift other elements
3. The element is a standard, expected part of the surface (not a contextual or rare action)

A tooltip explaining the disabled state is still required. Never use this to show controls the user can _never_ access (role/permission) — hide those.

## Quick reference

| Condition | State |
| --- | --- |
| Lacks permission/role, or feature not in tier | Hidden (unless a product decision to tease the upgrade). For nav items, see [`navigation.md`](./navigation.md) |
| Irrelevant to current context | Hidden |
| No discoverable path to enable | Hidden |
| System-generated / inherited / prior-step / pre-filled value | Read-only |
| Prerequisite visible on the page | Disabled + actionable tooltip or inline hint |
| High-frequency toolbar item, no proximate path | Disabled + tooltip (spatial-consistency exception only) |

## Worked example: role-based access on one detail page

A model detail page rendered for three roles. Run each element through the framework:

| Element | Admin | Editor | Viewer | Why |
| --- | --- | --- | --- | --- |
| **Delete** action (in the page-header action region) | Enabled | Hidden | Hidden | Q1: a non-admin can never delete → **hide**, don't disable. A permanently-disabled Delete is noise and leaks the capability. Gate the action region by role. |
| **Name / config** fields | Enabled | Enabled | Read-only | Viewer benefits from seeing values (Q1 yes) but they're informational (Q2 yes) → **read-only** (native `readonly` input or key-value display), never `disabled`. |
| **System ID / created date** | Read-only | Read-only | Read-only | System-generated for everyone → always read-only. |

**The rule for permissions/roles: hide, never disable.** Disabled is only for a missing _proximate_ prerequisite the user can satisfy now (a required field, a pending validation) — never for a restriction the role can never lift. For a viewer's editable-by-others field, the answer is read-only, not disabled.

Visibility is resolved **before** confirmation friction: only render a destructive action for an authorized role, _then_ apply the [`confirmations.md`](./confirmations.md) level. Hiding the control and confirming the click are two separate gates.
