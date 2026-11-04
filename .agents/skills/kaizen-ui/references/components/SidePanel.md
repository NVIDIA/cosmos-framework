# SidePanel

A `<dialog>` panel that slides in from the left or right edge to show supplementary content without leaving the current view.

## Notes

- Choose surface by context need: SidePanel when the task needs page context, Modal for brief focused tasks that do not need context, dedicated page for complex content needing the full viewport.

- Set `modal={true}` for forms or workflows that require commitment. `modal={false}`(default) is for browse/preview panels so users can keep interacting with the page (e.g. clicking other rows).

- Set `closeOnClickOutside={false}` for panels containing forms so users do not lose their input. Leave it at the default for previews.

- Always provide `slotHeading` so users know which context the panel represents.

- Order body content from highest to lowest importance so users can scan and understand quickly.

- For multi-dimensional status (e.g. Priority + State, Severity + Phase), render as a side-by-side labeled Badge pair at the top of the body — not as paragraph text or a single combined badge.

- Do not nest `<Panel>` blocks inside a SidePanel — nesting re-creates containment the SidePanel already provides and reads as visually heavy. Structure body content with section titles, key-value pairs, and spacing instead.

- Anti-patterns: complex layouts belong on a dedicated page; an always-visible split-view "side panel" built from a `<div>` or `<Panel>` is not the ephemeral SidePanel.

- List-detail integration defaults: `open` driven by selection state, `side="right"`, `modal={false}`, `closeOnClickOutside={true}`, `slotHeading` = selected item name. Pre-selecting an item creates an "always-on" pattern that confuses the toggle/close affordance.

- Keep the SidePanel mounted and drive visibility with `open`. Wrapping it in a conditional (`{isOpen && <SidePanel open={isOpen} />}`) unmounts the panel before its close animation can run, so the panel disappears instantly.

- SidePanel content is a preview, not the source of truth — always include a "View full details" escalation link or button to the dedicated detail page.

- Pass `bordered` when the panel has distinct header/footer sections — the divider gives visual separation between heading, content, and footer.

- Footer action ordering: primary action on the trailing (right) side, cancel on the leading (left) side — same as Modal.

- To set a custom width, set the `--side-panel-width` CSS variable(defaults to 320px).

- To render the panel relative to its parent instead of the viewport, set `relative={true}`.

## Examples

### Basic Side Panel

```tsx
<SidePanel
  slotTrigger={<Button>Open Panel</Button>}
  slotHeading="Details"
  slotFooter={
    <>
      <SidePanelCloseButton kind="tertiary">Cancel</SidePanelCloseButton>
      <Button color="brand">Save</Button>
    </>
  }
  style={{ top: "var(--nv-app-bar-height)" }}
>
  Panel content with supplementary information.
</SidePanel>
```

### Modal Side Panel

Use when the side panel content requires focused attention and should block interaction with the main page, for forms or workflows that require commitment.

```tsx
<SidePanel
  modal
  bordered
  slotNavigation={
    <Tabs
      items={[
        { children: "Tab A", value: "a" },
        { children: "Tab B", value: "b" },
        { children: "Tab C", value: "c" },
      ]}
    />
  }
  slotTrigger={<Button>Open Modal Panel</Button>}
  slotHeading="Important Information"
  slotFooter={
    <>
      <SidePanelCloseButton kind="tertiary">Close</SidePanelCloseButton>
      <Button color="brand">Acknowledge</Button>
    </>
  }
>
  This panel requires focused attention with a backdrop overlay.
</SidePanel>
```

### Form Side Panel

Use when the side panel wraps a form that needs native form submission behavior.

```tsx
<SidePanel
  bordered
  slotTrigger={<Button>Open Form</Button>}
  slotHeading="Create Resource"
  renderContent={({ children }) => <form onSubmit={handleSubmit}>{children}</form>}
  slotFooter={
    <>
      <SidePanelCloseButton kind="tertiary">Cancel</SidePanelCloseButton>
      <Button type="submit" color="brand">
        Create
      </Button>
    </>
  }
>
  Form fields go here
</SidePanel>
```

### Native Form Dialog Side Panel

Use when the panel hosts a server-action form and cancel must close without submitting. `formMethod="dialog"` lets the native dialog close without JavaScript.

```tsx
<SidePanel
  bordered
  slotTrigger={<Button>Edit Task</Button>}
  slotHeading="Edit Task"
  attributes={{
    SidePanelCloseButton: {
      type: "submit",
      formMethod: "dialog",
      formNoValidate: true,
    },
  }}
  renderContent={({ children }) => (
    <form action="/api/update-task" method="post">
      {children}
    </form>
  )}
  slotFooter={
    <>
      <input type="hidden" name="taskId" value="123" />
      <Button type="submit" formMethod="dialog" formNoValidate kind="tertiary">
        Cancel
      </Button>
      <Button type="submit" color="brand">
        Save Changes
      </Button>
    </>
  }
>
  Form fields go here.
</SidePanel>
```

### Multi Step Side Panel

Use a multi-step pattern for wizards, guided flows, or any task that progresses through discrete stages.

```tsx
() => {
  const [step, setStep] = useState(0);
  const totalSteps = 3;
  const isFirstStep = step === 0;
  const isLastStep = step === totalSteps - 1;
  return (
    <SidePanel
      bordered
      slotTrigger={<Button>Open Side Panel</Button>}
      slotHeading={`Step ${step + 1} of ${totalSteps}`}
      slotFooter={
        <>
          {!isFirstStep && (
            <Button kind="secondary" onClick={() => setStep((s) => Math.max(0, s - 1))}>
              Back
            </Button>
          )}
          <SidePanelCloseButton className="ml-auto" kind="tertiary">
            Cancel
          </SidePanelCloseButton>
          <Button color="brand" onClick={() => setStep((s) => Math.min(totalSteps - 1, s + 1))}>
            {isLastStep ? "Save" : "Next"}
          </Button>
        </>
      }
    >
      <Flex direction="col" gap="2">
        Content for step {step + 1}.
      </Flex>
    </SidePanel>
  );
};
```

### Composed

Use composed primitives when you need full control over the side panel layout and behavior.

```tsx
<SidePanelRoot>
  <SidePanelTrigger asChild>
    <Button>Open Panel</Button>
  </SidePanelTrigger>
  <SidePanelContent bordered>
    <SidePanelHeading>Composed Heading</SidePanelHeading>
    <SidePanelMain>Composed side panel content using primitives directly.</SidePanelMain>
    <SidePanelFooter>
      <SidePanelCloseButton kind="tertiary">Cancel</SidePanelCloseButton>
      <Button color="brand">Save</Button>
    </SidePanelFooter>
  </SidePanelContent>
</SidePanelRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ SidePanelTrigger?: NativeElementAttributes<"button", typeof SidePanelTrigger>; SidePanelAnchorTrigger?: NativeElementAttributes<"a", typeof SidePanelAnchorTrigger>; SidePanelDialog?: NativeElementAttributes<"dialog", typeof SidePanelDialog>; SidePanelHeading?: NativeElementAttributes<"div", typeof SidePanelHeading>; SidePanelNavigation?: NativeElementAttributes<"div", typeof SidePanelNavigation>; SidePanelMain?: NativeElementAttributes<"div", typeof SidePanelMain>; SidePanelFooter?: NativeElementAttributes<"div", typeof SidePanelFooter>; SidePanelCloseButton?: Omit<ButtonProps, "className" \| "kind" \| "color" \| "children"> }` | - | Native HTML attributes forwarded to the internal composed components. |
| bordered | `boolean` | `false` | Adds a divider between the body and the heading/footer. |
| children | `ReactNode` | - | Body content rendered between the heading and footer. |
| closeButtonProps | `Omit<ButtonProps, "className" \| "color" \| "children" \| "kind">` | - | Props forwarded to the built-in close button; set `type="submit"` with `formMethod="dialog"` for no-JS form patterns. |
| closeOnClickOutside | `boolean` | `true` | Whether clicking outside the panel closes it. Set to `false` for panels containing forms so users do not lose input. |
| defaultOpen | `boolean` | `false` | Initial open state when the dialog is uncontrolled. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces. |
| disabled | `boolean` | - | Disables the trigger. |
| forceMount | `boolean` | - | Ensures dialog content remains mounted when the dialog is not open. |
| hideCloseButton | `boolean` | `false` | Hides the built-in close button. Combine with `closeOnClickOutside={false}` and a handled `onEscapeKeyDown` to make the panel fully non-dismissible. |
| modal | `boolean` | `false` | When `true`, the dialog is opened via `HTMLDialogElement.showModal()`, rendering a top-layer backdrop and trapping focus. When `false`, the dialog is opened via `HTMLDialogElement.show()` as a non-modal dialog. |
| onEscapeKeyDown | `(event: KeyboardEvent) => void` | - | Fires when Escape is pressed inside the panel. Call `event.preventDefault()` to keep the panel open. |
| onInteractOutside | `(event: MouseEvent) => void` | - | Fires on any interaction outside the panel. Call `event.preventDefault()` to keep the panel open. |
| onOpenAutoFocus | `(event: Event) => void` | - | Fires when the panel opens, before focus moves to its first `autofocus` or focusable element. Call `event.preventDefault()` to manage focus yourself. |
| onOpenChange | `(open: boolean, request: { cancel: () => void; }) => void` | - | Callback invoked whenever the dialog requests an open state change, in both controlled and uncontrolled modes. A controlled dialog requests `false` before its exit animation, so keeping `open` as `true` keeps it open. When refusing, also call `request.cancel()` so a later close does not reuse the refused return value. Controlled prop changes made by the consumer are not echoed back through this callback. |
| onPointerDownOutside | `(event: MouseEvent) => void` | - | Fires on pointer events outside the panel. Call `event.preventDefault()` to keep the panel open. |
| open | `boolean` | - | Controlled open state. When provided, the hook operates in controlled mode and `defaultOpen` is ignored. The consumer is responsible for updating this value in response to {@link UseDialogOptions.onOpenChange} callbacks, and the dialog stays open until it becomes `false`. |
| portal | `boolean` | `false` | Renders the floating content in a React portal. |
| portalContainer | `HTMLElement` | `document.body` | Container for portaled floating content when no nearer portal target exists. |
| relative | `boolean` | `false` | Positions the panel absolutely within its closest positioned ancestor instead of fixed to the viewport. The ancestor must set `position: relative` and `overflow: hidden`. |
| renderContent | `(props: { children: React.ReactNode; }) => React.ReactElement<unknown, string \| React.JSXElementConstructor<any>>` | - | Render function that wraps the panel body, useful for injecting a `<form>` around the content and footer. |
| side | `"right" \| "left"` | `"right"` | Edge of the viewport the panel slides in from. |
| slotAnchorTrigger | `ReactNode` | - | Anchor-based trigger used for progressive enhancement when `id` is provided. Navigates to `fallbackHref` when JavaScript is unavailable. |
| slotFooter | `ReactNode` | - | Action area rendered at the bottom of the panel, typically primary and secondary buttons. |
| slotHeading | `ReactNode` | - | Title text rendered at the top of the panel. |
| slotNavigation | `ReactNode` | - | Region rendered between the heading and body for in-panel navigation such as tabs or breadcrumbs. |
| slotTrigger | `ReactNode` | - | Element rendered as the panel trigger; opens the panel when activated. |

## Exports

- Components: `SidePanel`, `SidePanelAnchorTrigger`, `SidePanelCloseButton`, `SidePanelContent`, `SidePanelDialog`, `SidePanelFooter`, `SidePanelHeading`, `SidePanelMain`, `SidePanelNavigation`, `SidePanelRoot`, `SidePanelTrigger`
- Hooks: `useSidePanelContext`
- Contexts: `SidePanelContext`
- Types: `SidePanelAnchorTriggerProps`, `SidePanelCloseButtonProps`, `SidePanelContentProps`, `SidePanelContextType`, `SidePanelDialogProps`, `SidePanelFooterProps`, `SidePanelHeadingProps`, `SidePanelMainProps`, `SidePanelNavigationProps`, `SidePanelProps`, `SidePanelRootProps`, `SidePanelTriggerProps`

## See also

- [Modal](./Modal.md)
- [Button](./Button.md)
