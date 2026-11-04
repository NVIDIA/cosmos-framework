# Modal

A `<dialog>` that interrupts the page to capture the user's attention until they act on or dismiss its content. Use for confirmations, focused tasks, and information the user must acknowledge before continuing.

## Notes

- Set `closeOnClickOutside={false}` when the modal contains a form so users do not lose their input by clicking the backdrop. Informational and confirmation modals can leave it at the default, which is already `false` when `alert` is set.

- Pick the surface by the choice's stakes: consequences (commits a setting, submits a form, deletes data) or short form → Modal; low-stakes refinement anchored to a control (date picker, column selector) → Popover; side-docked reference content → SidePanel; long or complex flow → dedicated page.

- If the content would require scrolling on a standard viewport, the workflow exceeds 3 steps, or any step has more than 5 fields, the modal is too constrained — move to a SidePanel or dedicated page. Use Modal for confirmations and short forms (1–5 fields).

- Render exactly one primary action in `slotFooter`, placed on the trailing edge. Place cancel or back actions on the leading edge using `ModalCloseButton kind="tertiary"`.

- Footer action pattern per modal type — Confirmation: Cancel tertiary + brand primary. Destructive confirmation (Delete, Terminate, Revoke): Cancel tertiary + danger primary. Form: Cancel tertiary + brand submit. Multi-step: Back secondary + Cancel tertiary pushed with `ml-auto` + Next brand.

- Set `alert` when the modal interrupts with an urgent message the user must answer, such as confirming a destructive or irreversible action. Put the message in the body, since it becomes the dialog's description, and keep a Cancel action in the footer. Leave `alert` off for forms and informational modals.

- Always provide a cancel or dismiss option unless `dismissible={false}` is intentional for a critical confirmation flow where accidental dismissal would be harmful.

- Reserve fullscreen modals for content-heavy presentations (previews, complex editors) — don't use fullscreen for simple confirmations.

## Examples

### Basic Modal

Use when you need to interrupt the user for a confirmation or brief task that requires focused attention.

```tsx
<Modal
  slotTrigger={<Button>Open Modal</Button>}
  slotHeading="Confirm Action"
  slotFooter={
    <>
      <ModalCloseButton kind="tertiary">Cancel</ModalCloseButton>
      <Button color="brand">Confirm</Button>
    </>
  }
>
  Are you sure you want to proceed with this action?
</Modal>
```

### Fullscreen Modal

Use when the modal content needs maximum screen real estate, such as previews or complex editors. Reserve fullscreen for content-heavy presentations — don't use it for simple confirmations.

```tsx
<Modal
  className="size-full"
  slotTrigger={<Button>Open Fullscreen</Button>}
  slotHeading="Fullscreen Modal"
  slotFooter={
    <>
      <ModalCloseButton kind="tertiary">Close</ModalCloseButton>
      <Button color="brand">Save</Button>
    </>
  }
>
  Content that benefits from maximum screen space.
</Modal>
```

### Non Dismissible Modal

Use when the user must complete an action before continuing, preventing dismissal via escape or backdrop click. Only set `dismissible={false}` for critical confirmation flows where accidental dismissal would be harmful — most modals should remain dismissible.

```tsx
<Modal dismissible={false} slotTrigger={<Button>Open Required Modal</Button>} slotHeading="Required Action" slotFooter={<Button color="brand">Acknowledge</Button>}>
  You must complete this action before continuing.
</Modal>
```

### Alert Modal

Set `alert` when the modal interrupts with an urgent message the user must answer, such as a destructive confirmation. It sets `role="alertdialog"`, uses the body as the accessible description, and ignores backdrop clicks while Escape and Cancel still dismiss it.

```tsx
<Modal
  alert
  slotTrigger={<Button color="danger">Revoke API Key</Button>}
  slotHeading="Revoke API key?"
  slotFooter={
    <>
      <ModalCloseButton kind="tertiary">Cancel</ModalCloseButton>
      <Button color="danger">Revoke Key</Button>
    </>
  }
>
  Applications using this key will stop working immediately. This cannot be undone.
</Modal>
```

### Multi Step Modal

Use for wizards or forms where accidental backdrop dismissal would lose user progress. Disables `closeOnClickOutside` and uses `ml-auto` on the cancel button to push it away from a left-aligned back button.

```tsx
<Modal
  closeOnClickOutside={false}
  slotTrigger={<Button>Configure Deployment</Button>}
  slotHeading="Step 2 of 3 — Resources"
  slotFooter={
    <>
      <Button kind="secondary">Back</Button>
      <ModalCloseButton className="ml-auto" kind="tertiary">
        Cancel
      </ModalCloseButton>
      <Button color="brand">Next</Button>
    </>
  }
>
  Choose the resources for this deployment. Your progress is preserved across steps.
</Modal>
```

### Form Modal

Use when the modal contains a form that should wrap the content for native form submission.

```tsx
<Modal
  slotTrigger={<Button>Open Form</Button>}
  slotHeading="Submit Feedback"
  renderContent={({ children }) => <form onSubmit={handleSubmit}>{children}</form>}
  slotFooter={
    <>
      <ModalCloseButton kind="tertiary">Cancel</ModalCloseButton>
      <Button type="submit" color="brand">
        Submit
      </Button>
    </>
  }
>
  Form content goes here
</Modal>
```

### Native Form Modal

Use for server-rendered actions that must work without JavaScript. The body is wrapped in a real `<form action>` while `formMethod="dialog"` on the cancel and X close buttons closes the modal without submitting.

```tsx
<Modal
  slotTrigger={<Button color="danger">Delete Item</Button>}
  slotHeading="Confirm Deletion"
  attributes={{
    ModalCloseButton: {
      type: "submit",
      formMethod: "dialog",
      formNoValidate: true,
    },
  }}
  renderContent={({ children }) => (
    <form action="/api/delete-item" method="post">
      {children}
    </form>
  )}
  slotFooter={
    <>
      <input type="hidden" name="itemId" value="123" />
      <Button type="submit" formMethod="dialog" formNoValidate kind="tertiary">
        Cancel
      </Button>
      <Button type="submit" color="danger">
        Delete
      </Button>
    </>
  }
>
  Are you sure you want to delete this item? This action cannot be undone.
</Modal>
```

### Anchor Trigger Modal

Use `slotAnchorTrigger` paired with `id` for progressive enhancement: JavaScript opens the modal in place, while users without JavaScript navigate to the anchor's `href` as a fallback page.

```tsx
<Modal
  id="modal-anchor-trigger-example"
  slotAnchorTrigger={<Anchor href="/settings">Open Settings</Anchor>}
  slotHeading="Settings"
  slotFooter={
    <>
      <ModalCloseButton kind="tertiary">Cancel</ModalCloseButton>
      <Button color="brand">Save</Button>
    </>
  }
>
  Adjust your preferences below.
</Modal>
```

### Composed

Use composed primitives when you need full control over the modal layout and behavior.

```tsx
<ModalRoot>
  <ModalTrigger asChild>
    <Button>Open Modal</Button>
  </ModalTrigger>
  <ModalDialog>
    <ModalContent>
      <ModalHeading>Composed Heading</ModalHeading>
      <ModalMain>Composed modal content using primitives directly.</ModalMain>
      <ModalFooter>
        <ModalCloseButton kind="tertiary">Cancel</ModalCloseButton>
        <Button color="brand">Save</Button>
      </ModalFooter>
    </ModalContent>
  </ModalDialog>
</ModalRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| alert | `boolean` | `false` | Presents the modal as an alert dialog, for an urgent message the user must respond to, such as confirming a destructive action. Sets `role="alertdialog"`, describes the dialog with its body content, and turns off backdrop-click dismissal unless `closeOnClickOutside` is set. Escape still dismisses it. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ ModalTrigger?: NativeElementAttributes<"button", typeof ModalTrigger>; ModalAnchorTrigger?: NativeElementAttributes<"a", typeof ModalAnchorTrigger>; ModalDialog?: NativeElementAttributes<"dialog", typeof ModalDialog>; ModalHeading?: NativeElementAttributes<"div", typeof ModalHeading>; ModalMain?: NativeElementAttributes<"div", typeof ModalMain>; ModalFooter?: NativeElementAttributes<"div", typeof ModalFooter>; ModalCloseButton?: Omit<ButtonProps, "className" \| "kind" \| "children"> }` | - | Native HTML attributes forwarded to the internal composed components. |
| closeButtonProps | `Omit<ButtonProps, "className" \| "children" \| "kind">` | - | Props forwarded to the built-in close button, useful for native form integration. |
| closeOnClickOutside | `boolean` | - | Whether the dialog can be closed by clicking outside the dialog. |
| defaultOpen | `boolean` | `false` | Initial open state when the dialog is uncontrolled. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces. |
| dismissible | `boolean` | `true` | Whether the modal can be closed by the user. When `false`, hides the close button and disables both escape-key and outside-click dismissal. |
| forceMount | `boolean` | - | Ensures dialog content remains mounted when the dialog is not open. |
| hideCloseButton | `boolean` | `false` | Hides the built-in close button without disabling other dismissal mechanisms (escape, outside click). |
| modal | `boolean` | `true` | When `true`, the dialog is opened via `HTMLDialogElement.showModal()`, rendering a top-layer backdrop and trapping focus. When `false`, the dialog is opened via `HTMLDialogElement.show()` as a non-modal dialog. |
| onEscapeKeyDown | `(event: KeyboardEvent) => void` | - | Called when Escape is pressed inside the modal. Call `event.preventDefault()` to keep the modal open. |
| onInteractOutside | `(event: MouseEvent) => void` | - | Called when any interaction occurs outside the dialog. Call `event.preventDefault()` to keep the modal open. |
| onOpenAutoFocus | `(event: Event) => void` | - | Called when the modal opens, before focus moves to its first `autofocus` or focusable element. Call `event.preventDefault()` to manage focus yourself. |
| onOpenChange | `(open: boolean, request: { cancel: () => void; }) => void` | - | Callback invoked whenever the dialog requests an open state change, in both controlled and uncontrolled modes. A controlled dialog requests `false` before its exit animation, so keeping `open` as `true` keeps it open. When refusing, also call `request.cancel()` so a later close does not reuse the refused return value. Controlled prop changes made by the consumer are not echoed back through this callback. |
| onPointerDownOutside | `(event: MouseEvent) => void` | - | Called when a pointer event occurs outside the dialog. Call `event.preventDefault()` to keep the modal open. |
| open | `boolean` | - | Controlled open state. When provided, the hook operates in controlled mode and `defaultOpen` is ignored. The consumer is responsible for updating this value in response to {@link UseDialogOptions.onOpenChange} callbacks, and the dialog stays open until it becomes `false`. |
| portal | `boolean` | `false` | Renders the floating content in a React portal. |
| portalContainer | `HTMLElement` | `document.body` | Container for portaled floating content when no nearer portal target exists. |
| renderContent | `(props: { children: React.ReactNode; }) => React.ReactElement<unknown, string \| React.JSXElementConstructor<any>>` | - | Wraps the inner content in a custom element (typically a `<form>`) so the modal participates in native submission flows. |
| slotAnchorTrigger | `ReactNode` | - | Anchor-based trigger that opens the modal on click and falls back to navigation when JavaScript is unavailable. |
| slotFooter | `ReactNode` | - | Footer content rendered below the body. Children are spaced automatically and right-aligned by default. |
| slotHeading | `ReactNode` | - | Heading rendered at the top of the modal. |
| slotTrigger | `ReactNode` | - | Trigger that opens the modal when activated. |

## Exports

- Components: `Modal`, `ModalAnchorTrigger`, `ModalCloseButton`, `ModalContent`, `ModalDialog`, `ModalFooter`, `ModalHeading`, `ModalMain`, `ModalRoot`, `ModalTrigger`
- Hooks: `useModalContext`
- Contexts: `ModalContext`
- Types: `ModalAnchorTriggerProps`, `ModalCloseButtonProps`, `ModalContentProps`, `ModalContextType`, `ModalDialogProps`, `ModalFooterProps`, `ModalHeadingProps`, `ModalMainProps`, `ModalProps`, `ModalRootProps`, `ModalTriggerProps`

## See also

- [SidePanel](./SidePanel.md)
- [Popover](./Popover.md)
- [Button](./Button.md)
