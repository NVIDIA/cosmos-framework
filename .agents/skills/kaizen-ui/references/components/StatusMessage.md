# StatusMessage

A full-area centered message displayed when there is no content to show — empty states, error pages, permission denied, not found. Replaces the content area rather than overlaying it.

## Notes

- The primary CTA in an empty-state StatusMessage goes in slotFooter and should use Button with kind="primary" color="brand".

- Use Notification (not StatusMessage) when content exists alongside the error. StatusMessage is for when there is genuinely nothing else to render.

- `slotHeading` should describe specifically what happened ("No clusters found", not "Nothing here"). `slotSubheading` explains why. `slotFooter` provides a specific verb+noun CTA ("Create cluster", not "Get started").

- Use `size="small"` for within-panel or within-card empty states. `size="medium"` (default) is for full-page states like 404s or no-results pages.

- StatusMessage replaces the content area rather than overlaying it — choose it when the surface is empty, use it for empty states, error pages, and permission-denied screens.. Use Toast/Notification for transient feedback while content is present, Banner for persistent page-level context above content, and Modal only when the user must respond before continuing. Avoid stacking these to prevent alert fatigue.

- Empty-state copy patterns: first-time empty → "No [items] yet" + "Create your first [item]"; search/filter empty → "No results found" + "Clear filters"; permission denied → "You don't have access" + "Request access"; error → "Something went wrong" + "Try again"; not found → "[Item] not found" + "Back to [parent]".

- Gotcha: slotFooter is wrapped in a row flex container with a gap. When adding multiple buttons, wrap them in a fragment to maintain the gap or use gap="inherit".

## Examples

### Basic Status Message

Use when a page or section has no content to display, such as empty search results or a blank state.

```tsx
<StatusMessage
  slotMedia={<Icon />}
  slotHeading="Forbidden"
  slotSubheading="You're not authorized to access this page."
  // use slotFooter when adding actionable next steps the user can take.
  slotFooter={
    <>
      <Button kind="tertiary">Back Home</Button>
      <Button color="brand">Request Access</Button>
    </>
  }
/>
```

### With Image Media Status Message

Use an illustration instead of an icon for empty states that benefit from a richer visual; size the image explicitly since `slotMedia` only auto-sizes SVGs.

```tsx
<StatusMessage
  slotMedia={<img src={PLACEHOLDER_IMG} alt="Empty team illustration" style={{ height: "160px", width: "160px" }} />}
  slotHeading="No Team Members Yet"
  slotSubheading="Start building your team by inviting colleagues."
  slotFooter={<Button color="brand">Invite</Button>}
/>
```

### Composed

```tsx
<StatusMessageRoot size="medium">
  <StatusMessageMedia>
    <Icon />
  </StatusMessageMedia>
  <StatusMessageHeader>
    <StatusMessageHeading>No Team Members Yet</StatusMessageHeading>
    <StatusMessageSubheading>Start building your team by inviting colleagues.</StatusMessageSubheading>
  </StatusMessageHeader>
  <StatusMessageFooter>
    <Button color="brand">Invite</Button>
  </StatusMessageFooter>
</StatusMessageRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **slotHeading** * | `ReactNode` | - | Primary message that names the status. Default `body/bold/2xl` or `xl` typography styles (based on size) |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ StatusMessageMedia?: NativeElementAttributes<"div", typeof StatusMessageMedia>; StatusMessageHeader?: NativeElementAttributes<"div", typeof StatusMessageHeader>; StatusMessageHeading?: NativeElementAttributes<"div", typeof StatusMessageHeading>; StatusMessageSubheading?: NativeElementAttributes<"div", typeof StatusMessageSubheading>; StatusMessageFooter?: NativeElementAttributes<"div", typeof StatusMessageFooter> }` | - | Native HTML attributes forwarded to the internal composed components. |
| size | `"small" \| "medium"` | `"medium"` | Visual scale of the message. - "small" - Use within constrained areas such as cards or side panels. - "medium" - Use for full-page empty states like 404s or no-results pages. |
| slotFooter | `ReactNode` | - | Action area rendered below the message, typically containing call-to-action buttons. |
| slotMedia | `ReactNode` | - | Icon or image rendered above the heading. SVG sizing and color are handled automatically. |
| slotSubheading | `ReactNode` | - | Supporting description rendered below the heading. Default `body/regular/md` typography styles |

`* = required prop`

## Exports

- Components: `StatusMessage`, `StatusMessageFooter`, `StatusMessageHeader`, `StatusMessageHeading`, `StatusMessageMedia`, `StatusMessageRoot`, `StatusMessageSubheading`
- Types: `StatusMessageFooterProps`, `StatusMessageHeaderProps`, `StatusMessageHeadingProps`, `StatusMessageMediaProps`, `StatusMessageProps`, `StatusMessageRootProps`, `StatusMessageSubheadingProps`
