# Popover

A floating panel that displays rich content, options, or actions anchored to a trigger.

## Notes

- For single-string hints, use Tooltip — click-to-reveal for simple text adds an unnecessary interaction step.

- The interaction test: Tooltip = hover + text-only, Popover = click + rich content, Dropdown = click + item selection, Modal = click + blocking. Use Popover only when the surface is click-triggered and the content is richer than plain text.

- Do not use Popover as a filter panel or to host complex multi-step forms — reach for Dropdown for action menus, or SidePanel for complex interactive content.

- Render the trigger as a Button or other clearly interactive element so users recognize the surface is clickable. Plain text triggers offer no affordance.

- Do not nest Popovers — flatten the information hierarchy instead.

- Use Popover for onboarding guidance and task walkthroughs (e.g. "Step 1 of 3: Configure your cluster") anchored to the relevant trigger.

- Use `align="start"` or `align="end"` when the popover content is wide enough to overflow the viewport if centered.

- Set `modal={true}` only when the content needs focus trapping (e.g. an inline form inside the popover).

- When the popover must work from server-rendered HTML or open before hydration, leave `open` unset so `PopoverTrigger` keeps its native `popovertarget`; `onOpenChange` can still observe state changes.

- To control an uncontrolled popover programmatically without losing native trigger behavior, attach a ref to `PopoverContent` and call `showPopover()` or `hidePopover()` on the content element.

- Top-level props are split by element kind: recognized `<button>` attributes go to `PopoverTrigger`; every other top-level prop (`className`, `onPointerEnter`, `data-*`, …) is spread onto `PopoverContent`. Use `attributes.PopoverTrigger` when something must target the trigger.

- Popover content must be self-contained — users should be able to understand and act on it without navigating elsewhere. Escalate to SidePanel or Modal if the content needs significant scrolling or multi-step interaction.

## Examples

### Basic Popover

Use when you need to show supplementary information triggered by a button click without navigating away.

```tsx
<Popover slotContent={<p>Additional context about this item.</p>}>
  <Button>More Info</Button>
</Popover>
```

### Positioning Popover

Combine `side` (`top` | `bottom` | `left` | `right`) and `align` (`start` | `center` | `end`) to control where the popover renders relative to its trigger. Override the defaults when the trigger sits near a viewport edge or sibling content would otherwise overlap.

```tsx
<Popover side="right" align="end" slotContent={<p>Right side, end aligned</p>}>
  <Button>Open</Button>
</Popover>
```

### With Chevron Popover

Add `AnimatedChevron` inside the trigger to give a visual hint that the button toggles an open/closed surface.

```tsx
<Popover slotContent={<p>Additional context about this item.</p>}>
  <Button kind="secondary">
    Open Popover
    <AnimatedChevron />
  </Button>
</Popover>
```

### With Rich Content Popover

Compose layout primitives inside `slotContent` to render rich, structured content like profile cards. Keep the content focused — avoid overcrowding the popover.

```tsx
<Popover
  side="right"
  align="start"
  slotContent={
    <Stack gap="density-md" className="w-[260px] p-4">
      <Stack direction="row" gap="density-md" align="center">
        <Avatar fallback="JD" size="large" />
        <Stack gap="density-sm">
          <Stack direction="row" gap="density-sm" align="center">
            <Text kind="label/bold/md">John Doe</Text>
            <Badge color="green" kind="solid">
              Active
            </Badge>
          </Stack>
          <Text kind="body/regular/sm">Product Designer</Text>
        </Stack>
      </Stack>
      <Stack direction="row" gap="density-sm">
        <Button kind="secondary" size="small">
          View Profile
        </Button>
        <Button kind="primary" color="brand" size="small">
          Message
        </Button>
      </Stack>
    </Stack>
  }
>
  <Button>View Team Member</Button>
</Popover>
```

### Modal Popover

Use when the popover content requires focused interaction and should prevent access to the rest of the page.

```tsx
<Popover
  modal
  slotContent={
    <div>
      <p>This popover traps focus and dims the background.</p>
      <Button>Action</Button>
    </div>
  }
>
  <Button>Open Modal Popover</Button>
</Popover>
```

### With Anchor Popover

Use `PopoverAnchor` when the popover should position against an element other than the trigger — for example, a row in a table where the action button lives in a different cell.

```tsx
<PopoverRoot>
  <Stack direction="row" gap="density-xl" align="center">
    <PopoverTrigger asChild>
      <Button>Trigger</Button>
    </PopoverTrigger>
    <PopoverAnchor>
      <Text kind="body/regular/md">Content anchors here</Text>
    </PopoverAnchor>
  </Stack>
  <PopoverContent>
    <p>This popover positions against the anchor, not the trigger.</p>
  </PopoverContent>
</PopoverRoot>
```

### Composed

Use composed primitives when you need full control over the popover trigger and content layout.

```tsx
<PopoverRoot>
  <PopoverTrigger asChild>
    <Button>Trigger</Button>
  </PopoverTrigger>
  <PopoverContent>
    <p>Composed popover content using primitives directly.</p>
  </PopoverContent>
</PopoverRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **slotContent** * | `ReactNode` | - | Content rendered inside the popover panel. |
| align | `"center" \| "start" \| "end"` | `"center"` | Alignment of the popover relative to its anchor. |
| asChild | `boolean` | - | Renders the popover panel into the element passed as `slotContent` instead of a `<div>`. |
| attributes | `{ PopoverTrigger?: NativeElementAttributes<"button", typeof PopoverTrigger> }` | - | Native HTML attributes forwarded to the internal composed components. |
| children | `ReactNode` | - | Trigger element that opens the popover. |
| closeOnFocusOutside | `boolean` | `true` | Closes the popover when focus moves from it, its trigger, or its anchor to an element outside all three. Applies to programmatic focus too, and has no effect on a modal popover.  "Outside" is decided by DOM containment, so a nested surface given its own `portalContainer` counts as outside and closes this popover. Leave nested content on the inherited portal, which mounts inside this popover. |
| defaultOpen | `boolean` | `false` | Initial open state when uncontrolled. |
| disabled | `boolean` | - | Disables the popover trigger, preventing it from opening the popover. |
| modal | `boolean` | `false` | Disables interaction with the rest of the page while the popover is open and hides it from assistive tech. |
| onCloseAutoFocus | `(event: Event) => void` | - | Called when focus returns to the trigger after the popover closes. Call `event.preventDefault()` to skip the default focus restoration. |
| onEscapeKeyDown | `(event: KeyboardEvent) => void` | - | Called when Escape is pressed inside the popover. Call `event.preventDefault()` to keep the popover open. |
| onFocusOutside | `(event: Event) => void` | - | Called when focus moves outside the popover, its trigger, and its anchor, while `closeOnFocusOutside` is enabled. Call `event.preventDefault()` to keep the popover open. The event's `target` is the newly focused element. |
| onInteractOutside | `(event: Event) => void` | - | Called after `onPointerDownOutside` when a pointer is pressed outside the popover. Focus moving outside calls `onFocusOutside` instead. Call `event.preventDefault()` to keep the popover open. |
| onOpenAutoFocus | `(event: Event) => void` | - | Called when focus first moves into the popover after it opens. Call `event.preventDefault()` to skip the default focus behavior. |
| onOpenChange | `(open: boolean) => void` | - | Called when the open state changes. |
| onPointerDownOutside | `(event: Event) => void` | - | Called when a pointer is pressed outside the popover. Call `event.preventDefault()` to keep the popover open. |
| open | `boolean` | - | Controlled open state. When provided, the consumer is responsible for updating it in response to `onOpenChange`. |
| portal | `boolean` | `false` | Renders the floating content in a React portal. |
| portalContainer | `HTMLElement` | `document.body` | Container for portaled floating content when no nearer portal target exists. |
| positionAnchor | `string` | - | Overrides the CSS anchor name the popover positions against. Use to pair with a custom anchor name set on `PopoverTrigger` or `PopoverAnchor`. |
| side | `"right" \| "bottom" \| "left" \| "top"` | `"bottom"` | Preferred side of the anchor to render against. Flips automatically when the popover would overflow the viewport. |

`* = required prop`

## Exports

- Components: `Popover`, `PopoverAnchor`, `PopoverContent`, `PopoverRoot`, `PopoverTrigger`
- Hooks: `usePopoverContext`
- Types: `PopoverAnchorProps`, `PopoverContentProps`, `PopoverContextType`, `PopoverProps`, `PopoverRootProps`, `PopoverTriggerProps`
