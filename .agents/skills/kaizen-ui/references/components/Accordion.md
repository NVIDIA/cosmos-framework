# Accordion

A vertically stacked group of sections the user can expand and collapse to show or hide their contents.

## Notes

- Prefer default single-open (i.e. not `multiple={true}`). Switch to `multiple` only when users genuinely need to compare or cross-reference content across sections at the same time.

- Use `defaultValue` to open the most important or most-used section on first render. If no section is clearly more important, leave them all collapsed so the user can scan the trigger labels first.

- Keep trigger labels concise — users skim them to choose a section. Detailed content belongs inside `slotContent`.

- Avoid nesting Accordions. If a section needs its own expand/collapse behavior, the information architecture is likely too deep — flatten it instead.

- If switching between (not collapsing) panels consider Tabs — Tabs show one panel at a time, Accordion can show multiple simultaneously.

- For an unstyled collapsible section to build your own styles on top of, use Collapsible.

- Avoid `multiple={true}` with long section content — it can create excessive page length; reconsider whether Accordion is the right pattern or whether the content should be on separate pages/tabs.

- If triggers need icons, place them consistently on the same side across all items.

## Examples

### Basic Accordion

By default the accordion allows for a single item to be open at a time. Set `multiple` to allow multiple items to be open at a time.

```tsx
<Accordion
  // defaults to single selection mode
  defaultValue="1"
  items={[
    {
      slotTrigger: "Item 1",
      slotContent: "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
      value: "1",
    },
    { slotTrigger: "Item 2", slotContent: "...", value: "2" },
  ]}
/>
```

### Default Open Multiple Items Accordion

```tsx
<Accordion
  defaultValue={["1", "2"]}
  items={[
    { slotTrigger: "Item 1", slotContent: "...", value: "1" },
    { slotTrigger: "Item 2", slotContent: "...", value: "2" },
  ]}
  multiple
/>
```

### With Icon In Trigger

Use gap="inherit" to use the Accordion defined gap for the icon and text.

```tsx
<Accordion
  items={[
    {
      slotTrigger: (
        <Flex align="center" gap="inherit">
          <Icon /> Item 1
        </Flex>
      ),
      slotContent: "...",
      value: "1",
    },
    {
      slotTrigger: (
        <Flex align="center" gap="inherit">
          <Icon /> Item 2
        </Flex>
      ),
      slotContent: "...",
      value: "2",
    },
  ]}
/>
```

### Composed

```tsx
<AccordionRoot>
  <AccordionItem value="1">
    <AccordionTrigger>Section 1</AccordionTrigger>
    <AccordionContent>Content for section 1</AccordionContent>
  </AccordionItem>
  <AccordionItem value="2">
    <AccordionTrigger>
      <Flex align="center" gap="inherit">
        <Icon />
        Section 2
      </Flex>
    </AccordionTrigger>
    <AccordionContent>Content for section 2</AccordionContent>
  </AccordionItem>
</AccordionRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `{ slotTrigger: ReactNode; slotContent: ReactNode; value: string; disabled?: boolean; chevronPosition?: "start" \| "end" }[]` | - | Section definitions rendered in order as the accordion's collapsible panels. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| collapsible | `boolean` | `true` | Allows the open item to be clicked to collapse itself; when `false`, the open item stays open until another item opens. Ignored when `multiple` is `true`. |
| defaultValue | `string \| string[]` | - | Item value to open initially when state is uncontrolled. |
| disabled | `boolean` | `false` | Disables the entire accordion and prevents all interactions. |
| multiple | `boolean` | `false` | Allows more than one item to be open at the same time. |
| name | `string` | - | Groups the underlying `<details>` elements under a shared name to enforce single-open behavior. Ignored when `multiple` is `true`. |
| onValueChange | `(value: string) => void \| (value: string[]) => void` | - | Called when the open item changes, with the new value or an empty string when all items close. |
| value | `string \| string[]` | - | Controlled value of the currently open item. Pair with `onValueChange` to manage state externally. |

`* = required prop`

## Exports

- Components: `Accordion`, `AccordionContent`, `AccordionItem`, `AccordionRoot`, `AccordionTrigger`
- Hooks: `useAccordionContext`, `useAccordionItemContext`
- Types: `AccordionContentProps`, `AccordionItemProps`, `AccordionProps`, `AccordionRootProps`, `AccordionRootPropsBase`, `AccordionTriggerProps`

## See also

- [Collapsible](./Collapsible.md)
- [Tabs](./Tabs.md)
