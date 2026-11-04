# Collapsible

An unstyled disclosure that toggles a single block of content between expanded and collapsed states. Use to hide secondary detail behind a clickable summary so users can reveal more on demand.

## Notes

- For a set of independently togglable sections that share a header treatment, prefer Accordion.

- The trigger must visually read as an interactive control — pair it with a chevron (e.g. `AnimatedChevron`) and ensure it has hover and focus affordances. Plain bold text reads as a section heading and users miss the interaction entirely.

- For a trigger with hover/focus affordances, render a `Button` with `asChild` around a `<div>`(do not want to nest interactive elements inside the trigger) and pass that as the `slotTrigger`. The Button supplies hover/focus/keyboard affordance, the chevron signals state, and the combination disambiguates the label from a heading.

- For a noun-phrase label like "Advanced parameters", wrap a `Button asChild kind="tertiary"><span>` in `slotTrigger` and add an `AnimatedChevron` so the affordance is unambiguous.

- Anti-patterns: do not render the trigger as a plain heading; do not hide the chevron behind hover-only visibility; do not use noun-phrase labels like "Advanced parameters" or "More information" without a chevron — the noun-phrase framing compounds the heading-look and users scan past it.

## Examples

### Basic Collapsible

Use when content should be hidden by default to reduce visual noise but available on demand.

```tsx
<Collapsible slotTrigger="Trigger content">Hidden content revealed on expand.</Collapsible>
```

### Default Open Collapsible

```tsx
<Collapsible defaultOpen slotTrigger="Hide details">
  Content visible by default.
</Collapsible>
```

### With Icon In Trigger

```tsx
<Collapsible
  slotTrigger={
    <span className="inline-flex items-center gap-2">
      <ChevronDown />
      Header
    </span>
  }
>
  Content
</Collapsible>
```

### With Chevron Trigger

The AnimatedChevron component automatically reads the open/closed state from Collapsible.

```tsx
<Collapsible
  slotTrigger={
    <>
      <AnimatedChevron /> Toggle content
    </>
  }
>
  Content
</Collapsible>
```

### With Button Trigger

When using a Button in slotTrigger with asChild, render it as a non-interactive element like <div> to avoid nested interactive elements, allowing the native <summary> element to handle the toggle behavior. The recommended pattern is `Button kind="tertiary"` with a trailing `AnimatedChevron`: the Button supplies hover/focus/keyboard affordance, the chevron signals state, and the combination disambiguates the label from a heading.

```tsx
<Collapsible
  slotTrigger={
    <Button asChild kind="tertiary">
      <div>Toggle content</div>
    </Button>
  }
>
  Content
</Collapsible>
```

### Composed

Use composed primitives when you want to render the trigger and content apart from each other.

```tsx
<CollapsibleRoot defaultOpen>
  <CollapsibleTrigger>
    <Button asChild kind="tertiary">
      <div>
        Toggle content
        <AnimatedChevron />
      </div>
    </Button>
  </CollapsibleTrigger>
  <CollapsibleContent>Additional content - visible by default.</CollapsibleContent>
</CollapsibleRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **children** * | `ReactNode` | - | Content rendered inside the expandable region. |
| **slotTrigger** * | `ReactNode` | - | Content rendered inside the trigger that toggles the region. |
| attributes | `{ CollapsibleTrigger?: NativeElementAttributes<"summary", typeof CollapsibleTrigger>; CollapsibleContent?: NativeElementAttributes<"div", typeof CollapsibleContent> }` | - | Native HTML attributes forwarded to the internal composed components. |
| defaultOpen | `boolean` | `false` | Initial open state when the component is uncontrolled. |
| disabled | `boolean` | `false` | Prevents the trigger from toggling the open state and blocks programmatic toggle commands. |
| onOpenChange | `(open: boolean) => void` | - | Called when the open state changes via user interaction or invoker commands. |
| open | `boolean` | - | Controlled open state. Pair with `onOpenChange` to manage state externally. |
| showChevron | `boolean` | `false` | Renders a chevron after the trigger content that rotates to reflect the open state. |

`* = required prop`

## Exports

- Components: `Collapsible`, `CollapsibleCommands`, `CollapsibleContent`, `CollapsibleRoot`, `CollapsibleTrigger`
- Hooks: `useCollapsibleContext`
- Contexts: `CollapsibleContext`
- Types: `CollapsibleContentProps`, `CollapsibleContextValue`, `CollapsibleProps`, `CollapsibleRootProps`, `CollapsibleTriggerProps`
