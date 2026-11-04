# Error states

Errors disrupt the user's task. The patterns job is to make the disruption as recoverable as possible — by picking the right scope (where the message lives), the right severity (how loud it is), and the right action (what the user can do about it).

## When to use

This pattern resolves four overlapping concerns: page-level error chrome, container-level "something went wrong", inline form validation, and transient action failures.

Use when:

- An operation failed (network request, validation, server-side action)
- A resource cannot be served (route gone, no access, system outage)
- An async operation reached a terminal failure state

Don't use when:

- The state is "no data yet" / empty result — that's an [empty state](./empty-states.md)
- The state is "still loading" — render a loading state, not an error
- The user is inside a successful flow with non-blocking warnings — use `Banner` `status="info"` or inline help text instead

---

## Anatomy

The content parts of an error display.

### Required

| Part | Purpose | Notes |
| --- | --- | --- |
| **Headline** | What didn't happen, in user-task terms | "Cluster restart didn't start." Not "Operation failed" or "Error 500." |
| **Affordance** | The action the user can take | Button or link; see [Modifiers → Affordance](#affordance) for which one. |

### Optional

Include when applicable.

| Part | Include when |
| --- | --- |
| **Explanation** | One sentence, plain language, what blocked the operation. **Required for terminal errors**; optional for transient ones. No stack traces. |
| **Error code** | See [Modifiers → Error code visibility](#error-code-visibility). If shown, place after the affordance, in muted text. |
| **Support link** | The user has exhausted self-service paths and the next step is a ticket. |

---

## Variants

Pick exactly one scope. The scope determines which component and which placement.

| Variant | Use when | Placement |
| --- | --- | --- |
| **Full-page** | The user cannot do anything on the page (404, 500, 403, account suspended) | Full-page error layout |
| **Page-level** | Partial outage/error affecting the entire view | `Banner` above layout content |
| **Container-level** | A specific region failed (panel, table, list, side-panel content) | `StatusMessage` rendered inside the failed container |
| **Inline (field-level)** | A field's value is invalid or rejected by the server | Directly under the offending field |
| **Transient (action result)** | An action the user just initiated failed (the page itself is fine; the user attempt is not) | `Toast` (recoverable) or `Notification` (needs user action) |

**Decision rule:** If the route itself is unavailable → full-page. If the user should be able to but can't do anything on this rendered page because of an outage/error → page-level. If a specific region failed → container-level. If a specific input is wrong → inline. If a specific action failed → transient.

**Mixed cases:** Multiple regions failed on a dashboard? Container-level for each region, optional summary banner only if the cumulative failure changes the page's value (e.g., 5 of 7 panels failed → the dashboard is mostly broken; show a banner _and_ in-panel errors).

### Transient sub-variants (component by severity)

When the user just initiated an action and it failed, pick the transient component by severity:

| Severity | Component | When |
| --- | --- | --- |
| **Recoverable, transient** | `Toast` `status="error"` with retry action | Single API call failed; retry inline solves it |
| **Persistent, requires user action** | `Notification` `status="error"` | Multiple retries failed; the user needs to make a decision (escalate? change inputs? wait?) |
| **System-wide impact** | `Banner` at top of `<main>` | The user's action exposed a system-level issue affecting other parts of the page |

**Do not auto-dismiss errors that need a decision.** Toasts auto-dismiss; that's only safe for errors without an action. Anything requiring a choice goes in `Notification` or `Banner`. See [`./feedback.md`](./feedback.md) for the broader Toast vs. Banner vs. Notification selection if you're still unsure.

---

## Composition rules

Apply to all variants.

### Voice: name what the user wanted, what didn't happen, what to do next

Errors are not the place for cleverness. They are the place for the truth, plus the next step. Keep the message brief.

| Do | Don't |
| --- | --- |
| "We couldn't reach the cluster." | "Your request failed." |
| "The cluster name is already taken — try another." | "Validation error: name field invalid." |
| "Restart didn't start. Try again, or check incidents." | "Something went wrong." |
| "You don't have access to this workspace. Switch accounts or request access." | "403 Forbidden" |

**The rule:** name what the user wanted, name what didn't happen, name what to do next. In that order.

### Never surface raw implementation details

Don't: Stack traces, HTTP method/URL, internal exception messages, framework error names.
Do: Render the human message; log the trace server-side or to console.

Exposing internals scares users, leaks attack surface, and offers no recovery path.

### Never show only an error code

The user has to translate it; most won't. Show the human message first, with the code (when shown) as a muted suffix.

### Match scope to impact

Don't: Page-level banner for a single panel's failure — implies the whole page is broken when only a part is.
Do: Container-level error in the failing panel; reserve the page banner for page-wide issues.

### Aggregate batch failures, don't stack toasts

Don't: A batch operation that emits one error toast per failed item — visual noise; user can't process them.
Do: One `Notification` with a count and a link to the affected items.

### Pick the affordance before writing the message

The error message resolves into exactly one of the affordances in [Modifiers → Affordance](#affordance). Pick first, then write the headline + explanation to support that affordance — not the other way around.

---

## Modifiers

### Recoverable vs. terminal

This drives which affordance you pick.

| State | Affordance | Examples |
| --- | --- | --- |
| **Recoverable** — user can retry from here | Retry button on the error itself | Network timeout, optimistic update conflict, transient 5xx, "we couldn't load this" |
| **Terminal** — user must leave | Navigation affordance (back, home, support) | 403 (no access), 404 (resource gone), account-suspended, locked-out auth |
| **Recoverable but elsewhere** | Link to where the user can fix it | "Quota exceeded — request more quota" (link to quota page); "Cluster locked by another deployment — view that deployment" |

A retry button on a terminal error sets a false expectation. A back-to-home button on a recoverable error gives up on the user's intent. Match the affordance to the state.

### Affordance

The error message resolves into exactly one of these affordances:

| Affordance | When | Anti-pattern |
| --- | --- | --- |
| **Retry** | Transient or recoverable failures (timeouts, conflicts, server hiccups) | Disabling retry on transient errors because "we don't know if it'll work." Retry costs the user nothing; not letting them retry costs them recovery. |
| **Navigate elsewhere** | Terminal errors, "the thing you wanted is somewhere else" | Sending the user to the home page when the back button would do |
| **Contact support** | Genuinely unrecoverable, opaque errors (system-wide outage, account-level issue) | Defaulting to "contact support" for transient errors — the user wants to retry, not file a ticket |
| **Wait** | Async operations the user can't influence (cluster provisioning, deployment in progress) | Showing an error when the operation is still in progress |
| **Acknowledge / dismiss** | Informational errors that don't require user action | Forcing dismiss on a notification the user already understands |

### Severity (within the chosen variant)

The component status dials tone within any variant. Match severity to the actual impact, not the loudness you want:

| Status | Use when |
| --- | --- |
| `status="error"` | The operation is fully broken; the user must act or leave |
| `status="warning"` | Partial / degraded; the page or operation is functional but reduced (e.g., "3 of 7 panels couldn't load") |
| `status="info"` | Informational; no user action required |

### Error code visibility

| Audience | Show error codes? |
| --- | --- |
| **Developer-facing apps** (CLI, SDK, console for ops) | Yes. Codes accelerate debugging and are part of the audience's vocabulary. Place after the human-readable message. |
| **End-user / customer-facing apps** | No by default. Codes confuse non-technical users without aiding recovery. **Exception:** show on a "show details" disclosure when contacting support is the affordance — the support agent will need it. |

Never show codes _instead of_ the human message. They go alongside, not in place of.
