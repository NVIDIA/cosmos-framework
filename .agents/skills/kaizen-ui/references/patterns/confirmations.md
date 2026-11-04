# Confirmations

A confirmation turns a click that _might_ have been accidental into one that is _certainly_ intentional — without slowing down actions that don't need it. Friction must match consequence: confirming everything trains users to dismiss confirmations reflexively; confirming nothing makes accidents irreversible.

## When to use

**The reversibility test is the gate.** If "oops" can be fixed by clicking Undo or re-running the action, it's reversible. If "oops" needs a support ticket or a backup restore, it's irreversible. Reversibility picks the level:

| Reversibility | Level |
| --- | --- |
| **Reversible in-session** (toggle a setting, archive a draft) | Level 0 — no modal, Toast with Undo |
| **Reversible with cleanup** (move to trash, deactivate, restart/pause shared infra) | Level 1 — confirm modal |
| **Irreversible** (permanently delete, terminate, revoke) | Level 2 — type-to-confirm modal |

**Impact names, it doesn't escalate.** Affecting shared infra or other users does _not_ raise the level — a reversible action stays reversible. Name the cross-user impact in the modal body or the success toast ("Cluster restart started — affects 4 active deployments") instead of forcing a stricter confirmation.

**Visibility comes first.** Only render a destructive action for a role allowed to perform it — gate the trigger per [`element-visibility.md`](./element-visibility.md) (lacks permission → hide, don't disable). Confirmation is a separate gate that applies _after_ the action is visible, not a substitute for authorization.

---

## Levels

Three levels of friction, plus two modifiers that layer onto Level 2.

| Level | Pattern | Friction | When |
| --- | --- | --- | --- |
| **0 — none** | Direct action + Toast with Undo | None | Reversible in-session |
| **1 — confirm** | Modal: Cancel + action button | One click | Reversible with cleanup |
| **2 — type-to-confirm** | Modal: type the entity name + Cancel + action | Type the name | Irreversible |

**Modifiers on Level 2:**

- **Bulk** — destructive action on many entities at once. Add a count + identifier list; type the count instead of a name. See [Bulk](#bulk-modifier).
- **Re-auth** — account-level, billing, or security-critical. Add a password / 2FA challenge before the action runs.

Don't skip levels in either direction: a type-to-confirm on a one-click toggle is over-friction; a bare toast on an irreversible bulk delete is under-friction.

---

## Anatomy

For any modal dialog (Level 1–2). Use [Modal](../components/Modal.md).

| Region | Content rule |
| --- | --- |
| **Heading** | Action verb + target. "Delete model 'llama-7b'?" not "Confirm deletion" or "Are you sure?" |
| **Body** | One sentence naming the consequence in user-task language: what's lost, what becomes inaccessible, what can't be undone. |
| **Cancel button** | Always present. Always says "Cancel" — not "No," not "Go back." |
| **Action button** | Verb + noun matching the heading: "Delete model." Color `danger` if destructive, else `brand`. |
| **Button order** | Trailing-primary: `[Cancel] [Action]` left-to-right. |
| **Initial focus** | **Cancel** for destructive actions; **Action** for non-destructive confirms. Default focus = the safer choice. |
| **Type-to-confirm input** (Level 2) | Button enables only on exact match. Show the expected value just above the input so the user can read-and-type. |

### Body style

The body answers: _what specifically am I losing if I click through?_ Name the concrete consequence, not a generic warning. If you can't write that sentence, the action probably doesn't need a confirmation — or you don't understand the consequence.

| Bad body | Good body |
| --- | --- |
| "This action cannot be undone." | "Deleting removes the model weights and unloads it from any active deployments. This can't be undone." |
| "Are you sure you want to do this?" | "All in-progress jobs will fail and queued jobs will be cancelled. Active users will be disconnected." |

---

## Composition rules

### Level 0 — Toast with Undo

The affordance is a [`Toast`](../components/Toast.md) `status="success"` with an Undo action ("Archived 3 jobs. Undo."). Undo reverses the action atomically; if reversal fails, the toast becomes an error with retry (see [`error-states.md`](./error-states.md)). Only actions with **no downstream side effects** are undoable — archiving locally is undoable; sending a message is not. After the toast dismisses, the action is permanent; don't surface a separate "undo last action" elsewhere.

### Type-to-confirm (Level 2)

| Question | Rule |
| --- | --- |
| **What does the user type?** | The entity's name (model, cluster, workspace). Not "DELETE." Not a UUID. The name is what the user already knows. |
| **Where is the expected string shown?** | Directly above the input, in `text-secondary` or a code-style background. Read-and-type, not memorize. |
| **What enables the button?** | Exact match. Case-sensitive. Whitespace-trimmed. See [`forms.md`](./forms.md) for input validation. |
| **What if there's no name?** | Use a stable identifier the user has seen. If none exists, reconsider whether type-to-confirm fits. |

**Why the name, not "DELETE":** typing a literal word doesn't prove the user understood _what_ they're deleting. Typing "llama-7b" forces a "yes, this specific thing" moment. That moment is the point.

### Bulk modifier

When a destructive action targets multiple selected entities:

| Aspect | Rule |
| --- | --- |
| **Heading** | Include the count: "Delete 5 clusters?" |
| **Body** | List the first 3 identifiers + "and N more"; for >10, show count and a link to the full list. |
| **Type-to-confirm** | Type the action verb + count: "delete 5". Entity names don't fit here. |
| **Action button** | "Delete 5 clusters" — match the count in the heading. |

Don't let the user deselect from inside the modal. Selection happened on the previous screen; going back to refine is the cancel path.

### Anti-patterns

| Anti-pattern | Why it's wrong | Do this instead |
| --- | --- | --- |
| Says "cannot be undone" but… can be | Erodes trust in real warnings | Be honest: "moves to trash; restorable for 30 days" |
| Disabling Cancel during a long-running action | User has no escape if something goes wrong | Cancel stays enabled; cancellation may be best-effort but must remain a path |
| Type-to-confirm on a reversible action because it "affects shared infra" | Reversibility is the gate; impact alone doesn't justify it | Stay at the reversible level; name the impact in the body or toast |

---

## Inline confirmation (no modal)

For low-stakes destructive actions in dense UIs (a row delete in a table), an inline two-step confirm can replace the modal:

```
[Delete] →click→ [Click again to confirm]  ↻ [Delete]
```

Use sparingly — only when:

- The action is reversible-with-cleanup, not truly irreversible
- The row's identity is visible (no navigating away to find it)
- The two-step button reverts after a 3–5 second timeout

**Never use inline confirm for Level 2 actions** — the button-replacement is too easy to dismiss accidentally for genuinely irreversible operations.

---

## Example

**Scenario:** User clicks "Delete model" on `llama-7b`. Deletion permanently removes weights and unloads from active deployments.

1. Reversibility → irreversible → **Level 2** (type-to-confirm).
2. The active deployments are cross-user impact → **named in the body**, not escalated. Single entity → no bulk modifier.
3. Modal:
   - Heading: "Delete model 'llama-7b'?"
   - Body: "Deleting removes the model weights and unloads it from any active deployments (currently 3). Active inference requests will fail. This can't be undone."
   - Type-to-confirm: input labeled "Type 'llama-7b' to confirm"; button disabled until exact match
   - Buttons: `[Cancel] [Delete model]`, trailing-primary, `danger` on Delete
