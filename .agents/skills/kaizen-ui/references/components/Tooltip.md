# Tooltip

Reveals a short, supplemental label when the user hovers or focuses an element. Use to clarify icon-only controls or surface non-essential context.

## Notes

- Pick the right primitive up front: Tooltip for a short, supplemental label on hover/focus (no interactive content); Popover for richer content with links or buttons that needs to stay open; FormField `slotInfo` for the canonical info-icon-next-to-form-label pattern.

- Wrap the application in a TooltipProvider so the open delay is skipped when the user moves between adjacent tooltips.

- Tooltips disappear on blur; never put interactive content, links, or critical information inside one. Use Popover instead.

- Tooltips are hidden until hover or focus — keyboard and mobile users may miss them. Do not hide essential information in a tooltip; surface it in visible labels or helper text instead.

- Keep tooltip content to 1-2 sentences. For longer or richer content (links, buttons, etc.), use Popover.

- Prefer `side="top"` or `side="bottom"` over `"left"`/`"right"` to avoid obscuring the trigger and adjacent content. Flip to the opposite side when the trigger sits near a viewport edge.

- Pass a keyboard-focusable trigger: a Button, an anchor, or a form control. Tooltip does not add a button around the child, so a child that cannot receive focus (Badge, Text, a bare icon) opens the tooltip on hover only and fails WCAG 2.1.1. Give it a button element instead, e.g. `<Badge asChild><button type="button">…</button></Badge>`.

- Any disabled element (not just `<Button disabled>`) may not fire focus or hover events consistently; wrap it in a focusable span when pairing it with a tooltip so the trigger still receives events.

- When an enabled button tooltip trigger needs its server-rendered click fallback, leave `open` unset so `TooltipTrigger` keeps its native `popovertarget`; disabled and composite triggers may not emit one, while `onOpenChange` can still observe state changes.

- For composite controls that split their props (Combobox, Popover), pass a render function as children and spread the provided props onto the control's anchor chrome (e.g. `attributes={{ InputShell: triggerProps }}`); a plain element child would otherwise route the tooltip's handlers onto the floating panel rather than the trigger. Simple controls (Dropdown, Select, plain buttons) can be passed directly as the element child.

## Examples

### Basic Tooltip

```tsx
<Tooltip slotContent="Helpful tooltip text">
  <button type="button">Hover me</button>
</Tooltip>
```

### Positioning Tooltip

Combine `side` (`top` | `bottom` | `left` | `right`) and `align` (`start` | `center` | `end`) to control where the tooltip renders. Override the defaults when the trigger sits near a viewport edge or when wider tooltip content would clip with centered alignment.

```tsx
<Tooltip align="start" side="bottom" slotContent="Start-aligned tooltip">
  <button type="button">Aligned start</button>
</Tooltip>
```

### Disabled Trigger Tooltip

Wrap a disabled Button to explain why the action is unavailable. Tooltip routes hover and focus through a focusable wrapper since disabled controls don't fire pointer or focus events.

```tsx
<Tooltip slotContent="You don't have permission to perform this action.">
  <Button disabled>Save</Button>
</Tooltip>
```

### Conditional Tooltip

Toggle the disabled prop to suppress the tooltip without unmounting it. Useful when the tooltip should only appear in certain runtime states (e.g. `disabled={!isReadOnly}`).

```tsx
<Tooltip disabled slotContent="Hidden while disabled is true">
  <Button>Always enabled</Button>
</Tooltip>
```

### With Icon Trigger Tooltip

Pair a tooltip with an icon-only button to surface an accessible label without consuming layout space. Always provide aria-label on the trigger so assistive tech still announces it.

```tsx
<Tooltip slotContent="Get help with this feature">
  <Button aria-label="Help" kind="tertiary" size="small">
    <InfoCircle />
  </Button>
</Tooltip>
```

### Max Width Tooltip

Apply a max-width utility via className when the tooltip content is long enough to wrap. Prevents the tooltip from stretching across the viewport.

```tsx
<Tooltip className="max-w-[200px]" slotContent="This is a longer tooltip message that demonstrates how to constrain the maximum width so multi-line content stays readable.">
  <button type="button">Long tooltip</button>
</Tooltip>
```

### With Provider Tooltip

Wrap your app in TooltipProvider to remove the hover delay when moving between adjacent tooltips.

```tsx
<TooltipProvider>
  <Tooltip slotContent="First tooltip">
    <button type="button">First</button>
  </Tooltip>
</TooltipProvider>
```

### Composed

Use composed primitives when you need full control over trigger and content rendering.

```tsx
<TooltipProvider>
  <TooltipRoot>
    <TooltipTrigger aria-label="Help">
      <InfoCircle />
    </TooltipTrigger>
    <TooltipContent>Composed tooltip content</TooltipContent>
  </TooltipRoot>
</TooltipProvider>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| align | `"center" \| "start" \| "end"` | `"center"` | Preferred alignment along the trigger edge. May shift to avoid viewport collisions. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ TooltipTrigger?: NativeElementAttributes<"button", typeof TooltipTrigger> }` | - | Native HTML attributes forwarded to the internal composed components. |
| children | `string \| number \| bigint \| React.ReactElement<unknown, string \| React.JSXElementConstructor<any>> \| Iterable<React.ReactNode> \| React.ReactPortal \| Promise<AwaitedReactNode> \| ((triggerProps: TooltipTriggerRenderProps) => ReactNode) \| boolean` | - | Trigger content, or a render function receiving {@link TooltipTriggerRenderProps} to spread onto a custom anchor element. Use the function form to route the trigger props onto a specific part of a composite control (e.g. Combobox's `InputShell`, Popover's `PopoverTrigger`) instead of merging them onto its root via `asChild`. |
| defaultOpen | `boolean` | `false` | Initial open state when uncontrolled. |
| disabled | `boolean` | - | Disables the tooltip so it never opens. Useful for conditionally suppressing a tooltip without unmounting it. |
| id | `string` | - | Stable identifier used to derive the tooltip's content id and CSS anchor name. Provide when SSR ids must match between server and client. |
| onEscapeKeyDown | `(event: KeyboardEvent) => void` | - | Called when the Escape key is pressed while the tooltip is open. Call `event.preventDefault()` to keep it open. |
| onOpenChange | `(open: boolean) => void` | - | Called when the open state changes. |
| onPointerDownOutside | `(event: Event) => void` | - | Called when a pointer event occurs outside the tooltip. Call `event.preventDefault()` to keep it open. |
| open | `boolean` | - | Controlled open state. Pair with `onOpenChange`. |
| openDelayDuration | `number` | `100` | Delay in milliseconds from when the pointer enters a trigger to when the tooltip opens. |
| portal | `boolean` | `false` | Renders the floating content in a React portal. |
| portalContainer | `HTMLElement` | `document.body` | Container for portaled floating content when no nearer portal target exists. |
| positionAnchor | `string` | - | CSS anchor name this content positions against. Pair with the matching `anchorName` on TooltipTrigger when wiring tooltips across server boundaries or detached DOM trees. |
| side | `"right" \| "bottom" \| "left" \| "top"` | `"top"` | Preferred side of the trigger to position the tooltip on. May flip when there is not enough space. |
| slotContent | `ReactNode` | - | Tooltip content. Keep it short; consider constraining the width to preserve readability. |

## Exports

- Components: `Tooltip`, `TooltipContent`, `TooltipProvider`, `TooltipRoot`, `TooltipTrigger`
- Hooks: `useTooltipContext`
- Contexts: `TooltipProviderPresenceContext`
- Types: `TooltipContentProps`, `TooltipContextType`, `TooltipProps`, `TooltipProviderProps`, `TooltipRootProps`, `TooltipTriggerProps`, `TooltipTriggerRenderProps`

## See also

- [Popover](./Popover.md)
- [FormField](./FormField.md)
