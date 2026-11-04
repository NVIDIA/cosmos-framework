# Error Page (Layout Reference)

For full-page error states where the user cannot do anything on the requested route. Used for 404 (page not found), 500 (server error), 403 (forbidden), 401 (sign-in required), and similar terminal failures.

For non-full-page errors — region failures, page-level banners, inline validation, transient action failures — see [error-states.md](../patterns/error-states.md) → Decision 1 (scope). This layout is only for the **full-page** scope.

## When to use

- The route itself is invalid, gone, or inaccessible (404, 410)
- The user is not authorized to view the resource (403, 401)
- The server cannot serve any content for this route right now (500, 503, terminal outage)
- The user must leave this URL to recover — retry-in-place won't help

**Not for:**

- A panel or table that failed to load while the rest of the page works → [error-states.md](../patterns/error-states.md) Decision 1 → container-level
- A page-level outage where chrome and other regions remain functional → [error-states.md](../patterns/error-states.md) Decision 1 → page-level banner
- Form-field validation → [error-states.md](../patterns/error-states.md) Decision 1 → inline
- An action the user just initiated that failed → [error-states.md](../patterns/error-states.md) Decision 1 → transient

## Region rules

Error pages should still be rendered within the page shell.

### Main — centered both axes

Ensure content fills the remaining viewport vertically and centers in both axes. Cap the `StatusMessage` region at 640px.

## Anti-patterns (KUI-specific)

| Anti-pattern                     | Why it's wrong                                          | Do this instead                                                |
| -------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------- |
| Standalone "Back" button         | A loose button competes with the primary affordance     | Use an inline home link, or one primary sign-in button for 401 |
| Search input on 403 or 5xx       | Search won't change permissions or fix the server       | Omit search on those variants                                  |
| Auto-redirecting after N seconds | Users may not have read the message; takes control away | Static page; let the user choose                               |

---

## Variants

| Status | Heading              | Body                                                                                              | Primary action                                 |
| ------ | -------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| 404    | Page not found       | The page you're looking for doesn't exist or has been moved.                                      | Native "Back to home" link; search is optional |
| 403    | Access denied        | You don't have permission to view this resource. Request access if you believe this is a mistake. | Native "Back to home" link                     |
| 401    | Sign in required     | You need to be signed in to view this page.                                                       | Primary "Sign in" button                       |
| 500    | Something went wrong | We hit an unexpected error. Try again, and if the problem persists, contact support.              | "Try again" button                             |

## React examples

### Page not found

Error-page body inside the stacked shell (`max-w-7xl` main) from layout-principles.md: centered status code and 640px StatusMessage, plus an optional 420px search input for 404-style discovery failures.

```tsx
<Stack gap="6" align="center" className="w-full">
  <StatusMessage
    className="w-full max-w-[640px]"
    slotMedia={<span className="font-bold text-primary">404</span>}
    slotHeading="Page not found"
    slotSubheading={
      <span className="text-secondary">
        The page you were looking for could not be found.
        <br />
        You can return to the <Anchor href="/">homepage</Anchor>, search, or <Anchor href="/help">get help</Anchor> if you can't find what you're looking for.
      </span>
    }
  />
  <TextInput type="search" slotStart={<Icon name="magnifying-glass" />} placeholder="Search..." className="w-full max-w-[420px]" aria-label="Search the site" />
</Stack>
```
