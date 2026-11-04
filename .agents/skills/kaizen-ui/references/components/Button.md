# Button

A clickable element that triggers an action. Use specific verb + noun labels instead of vague text.

## Notes

- Unlike native <button>, you will need to set type="submit" for it to act as a form submit button.

- For page-level CTAs, PageHeader slotActions, empty-state CTAs, and modal primary actions, use color="brand".

- For destructive actions (Delete, Terminate, Revoke), use kind="primary" color="danger" and pair with a confirmation Modal.

- Status-changing actions that can be reversed (Decline, Reject, Archive, Dismiss) use kind="secondary" color="neutral", not color="danger". Danger styling on the secondary action fights the primary for attention; if the action can be undone or reversed, it is not destructive.

- Only one kind="primary" button per view (page, dialog, panel). Demote competing actions to secondary or tertiary (Hick's Law).

- Escalation path is Tertiary → Secondary → Primary. If a tertiary action is being missed, promote it to secondary — do not jump straight to primary.

- "Clear all" / "Clear filters" buttons should use kind="tertiary" and be placed alongside the filter chips they reset, not in the discovery toolbar.

- Stack order is context-dependent: commitment contexts (dialogs, form footers, page-level action bars) use `[tertiary] [secondary] [primary]` with primary trailing; CTA contexts (hero stacks, empty states, onboarding) use `[primary] [secondary] [tertiary]` with primary leading.

- For single action → Button; 2-4 related actions → ButtonGroup; navigation to URL → `Button asChild` wrapping `<a>`; inline prose navigation → Anchor.

- Use specific verb + noun labels ("Create cluster", "Delete workspace"); avoid generic verbs and keep to 2-3 words. If the action requires more than 3 words to describe, the surrounding UI should provide that context instead.

- Use `loading` for an action the user just triggered (submitting, saving).

- Guard repeated submissions in the form's submit handler; loading only blocks this button's activation after hydration.

- Native link destinations, submit types, and popover targets remain unchanged; guard those actions in your application, including before hydration.

- Custom children must forward props and refs, render children directly inside their root element, and guard handlers they own.

- For a single-button file upload, use `asChild` on a `<label>` that wraps `UploadInputElement` inside `UploadRoot`. Set `disabled` on `UploadRoot` rather than `loading` on the Button, since a label has no busy or disabled state of its own.

- Use children for the label in both states; `slotLoading` customizes only the indicator, including with `asChild`.

- For a custom arrow indicator, pass `<Spinner size="small" aria-label="Loading" />` to `slotLoading`. The slot makes it decorative; announce progress in a live region outside the button.

- Prefer an ellipsis (`...`) at the end of loading labels, such as "Sending..."; this is writing guidance, not automatic punctuation.

- Icon-only buttons must set `aria-label` describing the action. Reserve icon-only buttons for universally understood symbols (close, edit, delete, settings, kebab).

- Tertiary buttons need an affordance so they read as interactive. Options: trailing arrow/right caret for navigation-style actions, leading icon for utility verbs ("Edit", "Configure"), a color shift, or uppercase + letterspacing for dev-tool UIs. The default for hero contexts is the trailing arrow. A tertiary button with no icon, no color shift, and no label styling reads as a caption rather than an action.

## Examples

### Basic Button

```tsx
<Flex direction="col" gap="2">
  <Button kind="primary" color="brand">
    Button
  </Button>
  <Button kind="secondary" color="neutral">
    Button
  </Button>
  <Button color="danger">Button</Button>
</Flex>
```

### With Icon Button

```tsx
<Button>
  <Document />
  Add Document
</Button>
```

### Icon Only Button

```tsx
<Button aria-label="Add document">
  <Document />
</Button>
```

### Primary Action Button

Preferred default for primary actions: primary & brand. Use the primary action button for the most important action in a context. It should be used sparingly and only for the most important actions(submit, create, deploy, save, etc)

```tsx
<Button color="brand">Create cluster</Button>
```

### Secondary Action Button

Preferred default for secondary actions: secondary & neutral. Use the secondary action button for actions that are not the most important in a context. It should be used for actions that are not the most important in a context(cancel, close, etc)

```tsx
<Button kind="secondary">Deploy to GPU</Button>
```

### Tertiary Or Inline Action Button

Preferred default for tertiary/inline actions: tertiary & neutral. For example - edit, configure, kebab menu triggers, etc

```tsx
<Button kind="tertiary">View logs</Button>
```

### Destructive Action Button

Preferred default for destructive actions: primary & danger. Use the destructive action button for actions that are destructive and cannot be undone. For status-changing or reversible actions (remove from list, unassign, archive), use secondary + neutral. Reserve `color="danger"` for irreversible destructive actions.

```tsx
<Button color="danger">Delete cluster</Button>
```

### Button As Link

```tsx
<Button asChild>
  <a href="https://example.com">Link</a>
</Button>
```

### Button Loading Methods

Loading buttons with an unchanged label, a changing label, an icon-only action, no indicator, and a custom arrow spinner.

```tsx
<Flex direction="col" align="start" gap="4">
  <Flex direction="col" align="start" gap="2">
    <span>Same text, default spinner</span>
    <Button loading>Send</Button>
  </Flex>
  <Flex direction="col" align="start" gap="2">
    <span>Changing text, default spinner</span>
    <Button loading>Sending...</Button>
  </Flex>
  <Flex direction="col" align="start" gap="2">
    <span>Icon only</span>
    <Button loading aria-label="Send message" />
  </Flex>
  <Flex direction="col" align="start" gap="2">
    <span>Changing text, no spinner</span>
    <Button loading slotLoading={null}>
      Sending...
    </Button>
  </Flex>
  <Flex direction="col" align="start" gap="2">
    <span>Same text, no spinner</span>
    <Button loading slotLoading={null}>
      Send
    </Button>
  </Flex>
  <Flex direction="col" align="start" gap="2">
    <span>Custom arrow spinner</span>
    <Button loading slotLoading={<Spinner kind="default" size="small" className="[--spacing:2px]" aria-label="Sending" />}>
      Sending...
    </Button>
  </Flex>
</Flex>
```

### Button With Truncation

Button text should be short and concise, and if not text will wrap to the next line to ensure users can read the text. If you prefer to truncate the text you should set the `title` prop.

```tsx
<Button title="Download the full results as a CSV file">
  <span className="truncate">Download the full results as a CSV file</span>
</Button>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| color | `"brand" \| "neutral" \| "danger"` | `"neutral"` | The color variant of the button. - "brand" - Prominent calls-to-action such as page-level CTAs and modal primary actions. - "neutral" - Regular actions, suitable for most use cases. - "danger" - Destructive actions that cannot be undone, such as delete or remove. |
| disabled | `boolean` | - | Disables the button. |
| kind | `"primary" \| "secondary" \| "tertiary"` | `"primary"` | The kind of button. - "primary" - The most important call-to-action on the page. Only one per context. - "secondary" - Regular actions, suitable for most use cases. - "tertiary" - Low-priority or supplemental actions. |
| loading | `boolean` | `false` | Shows a loading indicator alongside children and applies the pressed treatment while retaining focus and reporting `aria-busy` and `aria-disabled`. Suppresses activation events after hydration; guard native navigation, form submission, and custom child handlers in your application. |
| size | `"small" \| "tiny" \| "medium" \| "large"` | `"medium"` | The size of the button. - "large" - The main call-to-action for a page or section. - "medium" - Suitable for most use cases. - "small" - Compact layouts with limited space or less significant actions. - "tiny" - Dense layouts such as table cells where horizontal space is at a premium. |
| slotLoading | `ReactNode` | - | Noninteractive inline loading indicator; omit or pass `undefined` for the default spinner, or pass `null` for no indicator. Keep loading text in children. With `asChild`, render children directly inside the child's root element. Void elements such as `<input>` cannot contain a custom indicator, so `slotLoading` is ignored for them. |

## Exports

- Components: `Button`
- Types: `ButtonProps`

## See also

- [ButtonGroup](./ButtonGroup.md)
- [Anchor](./Anchor.md)
