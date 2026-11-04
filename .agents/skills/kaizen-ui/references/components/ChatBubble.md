# ChatBubble

The surface of one chat message: a padded, rounded block for a person's words, or flat text for an AI reply.

## Notes

- Pick `kind` by author: `"strong"` for the current user, `"subtle"` (the default) for another person, `"flat"` for an AI or system reply, which drops the surface and padding so long text reads like a document.

- The bubble is fluid: it fills its container and sets no width, max width, height, or alignment of its own. The layout around it, such as a Stack or Flex, owns all of that.

- Children are the content: usually text, but anything renders. When a reply asks for a decision, pass the `Button`s to `slotActions` so they render last inside the bubble, after the text, and read as part of the reply. `slotActions` also works with `asChild`. Compose `ChatBubbleActions` yourself only when the row must sit somewhere other than last. It sets the space above the row and the gap between items, so do not wrap the actions in a `Flex` or add margins. It has no role; add `role="group"` and an `aria-label` through `attributes.ChatBubbleActions` only when the actions need a name.

- The bubble has no role. Give a transcript container `role="log"` and an `aria-label`, and put the sender's name beside the bubble as plain text, not a heading.

- Use `asChild` to render the bubble as another semantic element when needed.

- ChatBubble is for conversation turns. Use Notification or Toast for system feedback and Card for an entity.

## Examples

### Basic Chat Bubble

The default `kind="subtle"` is another person: a padded bubble on the highlight surface. The bubble fills its container, so the container decides how wide a message can be.

```tsx
<ChatBubble>Can you summarize last night's deployment?</ChatBubble>
```

### Agent Chat Bubble With Actions

`kind="flat"` is an AI or system reply: flat text with no surface or padding, so long generated text reads like a document. Content is any ReactNode, so when the reply asks for a decision, pass the Buttons to `slotActions`. They render in a row after the text, spaced from the text and from each other, so they read as part of the reply.

```tsx
<ChatBubble
  kind="flat"
  slotActions={
    <>
      <Button kind="secondary" size="small">
        Retry now
      </Button>
      <Button kind="tertiary" size="small">
        Wait for the window
      </Button>
    </>
  }
>
  Three services rolled out overnight. Two succeeded and one was held back after a failed health check. Should I retry the failed rollout now or wait for the next maintenance window?
</ChatBubble>
```

### Chat Bubble Conversation

One way a consumer lays out a back-and-forth conversation. The bubble is fluid and places nothing itself, so everything here beyond the bubbles is the consumer's layout: a Flex row per turn with the Avatar bottom-aligned beside the bubble, `justify="end"` to put the current user on the right, and a Stack capped at 70% so short messages hug their text while long ones wrap. The closing `kind="flat"` turn is the assistant: flat text with no cap, so a long reply reads like a document. The sender's name is plain text beside each bubble, and the transcript carries `role="log"` and an `aria-label` so it is a named live region.

```tsx
<Stack aria-label="Conversation with Jordan Lee" gap="4" role="log">
  <Flex align="end" gap="2" justify="end">
    <Stack align="end" gap="1" style={{ maxWidth: "70%" }}>
      <Text kind="label/bold/md">You</Text>
      <ChatBubble kind="strong">How did last night's deployment go?</ChatBubble>
    </Stack>
    <Avatar alt="" aria-hidden="true" fallback="You" />
  </Flex>
  <Flex align="end" gap="2">
    <Avatar alt="" aria-hidden="true" fallback="JL" />
    <Stack gap="1" style={{ maxWidth: "70%" }}>
      <Text kind="label/bold/md">Jordan Lee</Text>
      <ChatBubble>Two of the three services came up clean. The third was held back after a failed health check on its canary, so nothing unhealthy reached production.</ChatBubble>
    </Stack>
  </Flex>
  <Flex align="end" gap="2" justify="end">
    <Stack align="end" gap="1" style={{ maxWidth: "70%" }}>
      <Text kind="label/bold/md">You</Text>
      <ChatBubble kind="strong">Should we retry it now?</ChatBubble>
    </Stack>
    <Avatar alt="" aria-hidden="true" fallback="You" />
  </Flex>
  <Flex align="end" gap="2">
    <Avatar alt="" aria-hidden="true" fallback="JL" />
    <Stack gap="1" style={{ maxWidth: "70%" }}>
      <Text kind="label/bold/md">Jordan Lee</Text>
      <ChatBubble>Let's wait for the maintenance window. I'm looking at the failed canary now.</ChatBubble>
    </Stack>
  </Flex>
  <Flex align="end" gap="2">
    <Avatar alt="" aria-hidden="true" fallback="AI" />
    <Stack gap="1">
      <Text kind="label/bold/md">Assistant</Text>
      <ChatBubble kind="flat">
        The canary failed on tail latency, not on errors. A retry would rerun the same checks against the same traffic, so waiting for the window is the safer call. I can prepare the retry so it is
        one click when the window opens.
      </ChatBubble>
    </Stack>
  </Flex>
</Stack>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ ChatBubbleActions?: NativeElementAttributes<"div", typeof ChatBubbleActions> }` | - | Native HTML attributes forwarded to the internal composed components. |
| kind | `"strong" \| "flat" \| "subtle"` | `"subtle"` | The bubble's surface. - "strong" - The emphasis surface, for the current user. - "subtle" - The highlight surface, for another person. - "flat" - Flat text with no surface or padding, for an AI or system reply. |
| slotActions | `ReactNode` | - | Actions rendered in a `ChatBubbleActions` row after the content, such as the Buttons for a decision the reply asks for. With `asChild`, the row renders inside the child element, so the child must accept block content, such as an `<li>` or `<article>`, not a `<p>` or `<span>`. |

## Exports

- Components: `ChatBubble`, `ChatBubbleActions`
- Types: `ChatBubbleActionsProps`, `ChatBubbleProps`

## See also

- [Stack](./Stack.md)
- [Avatar](./Avatar.md)
