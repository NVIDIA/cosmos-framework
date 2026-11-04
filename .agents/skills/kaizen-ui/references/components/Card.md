# Card

A container that groups content, actions, and optional media about a single subject. Cards are fluid and grow to fill their container.

## Notes

- The Card already has padding, so do not add padding to the Card component or to it's children.

- Card is for single-subject entity display (one cluster, one user, one model). Do not use Card as a general container on dashboards — use Panel instead. Card's interaction states and internal structure add unintended visual layering when used as a generic wrapper.

- Cards are fluid and grow to fill their container — do not set fixed widths. Control sizing through the parent layout (e.g. a responsive Grid). Apply `className="h-fit"` when a card should shrink to its content height.

- Identity test for Card vs Panel: if the container represents "a thing" with its own name and attributes, use Card; if it represents "a region of content", use Panel.

- Pair `selected` with a visible affordance for what selection means — a bulk-action toolbar, a selection-count indicator, or a primary action — so users understand the consequence of selecting a card.

- A Card is EITHER a single click target (set `interactive` and render via `asChild` as a link, button or label with containing hidden input) OR carries inline action buttons on its surface — never both. Combining them creates competing click targets and ambiguous focus order.

- When a Card is `interactive`, do not place other interactive elements (Button, Anchor, Menu, etc.) inside it. Nested click targets break the "whole card is one click" affordance.

## Examples

### Basic Card That Fits Its Content

```tsx
<Card className="h-fit">
  <Badge kind="solid" color="gray">
    Badge
  </Badge>
  <Text kind="body/bold/2xl">Header</Text>
  <Text kind="body/regular/md">Lorem ipsum dolor sit amet</Text>
</Card>
```

### With Header Card

The header slot is rendered absolutely over the media. Use it to add a persistent label or badge above the body content. If there is no media, the header will be rendered above the body content. It is a flex container with a preset gap.

```tsx
<Card
  slotHeader={
    <Flex gap="inherit">
      <Badge>New</Badge>
      <Badge>Alt</Badge>
    </Flex>
  }
>
  Card body content
</Card>
```

### With Media Card

Use when the card needs a visual hero area like an image or video above or alongside the content. If the media rendered is dark or light, and you're using `slotHeader` in conjunction with it, you may want to set `mediaTheme` to ensure text and components are readable. This will set light/dark theme in the header slot to ensure sufficient contrast.

```tsx
<Card slotHeader={<span>Featured</span>} slotMedia={<MediaImg />} mediaTheme="light">
  Card content below media
</Card>
```

### Model Card Example

Canonical browse-collection card anatomy for a single entity (model catalog, dataset gallery, template library, related-item strip). Slot order: header `Badge` (resource type) → publisher → title → description → topic `Tag` row (`outline` / `gray`) → footer stats (`text-placeholder`, left-aligned, never `justify="between"`). Every card in the collection must share this same slot structure — see the entity-cards pattern guide.

```tsx
<Card slotHeader={<Badge kind="solid">Model</Badge>}>
  <Flex direction="col" gap="density-sm">
    <Text className="text-secondary" kind="label/regular/sm">
      NVIDIA
    </Text>
    <Text kind="body/bold/xl">Nemotron 3 Super 120B</Text>
    <Text kind="body/regular/md">120B-parameter reasoning model optimized for enterprise RAG.</Text>
  </Flex>
  <Flex gap="2" wrap="wrap">
    <Tag kind="outline" color="gray" readOnly>
      Reasoning
    </Tag>
    <Tag kind="outline" color="gray" readOnly>
      English
    </Tag>
    <Tag kind="outline" color="gray" readOnly>
      RAG
    </Tag>
  </Flex>
  <Text className="text-placeholder" kind="label/regular/sm">
    Updated 3d · 12.4k · 120B
  </Text>
</Card>
```

### Interactive Card

Use when the entire card should be a clickable target, such as navigation or selection. Render as an `<a>` for navigation or a button for actions. Do not place other interactive elements (Button, Anchor, Menu, etc.) on the card surface — competing click targets are ambiguous. If you need inline actions per card instead, omit `interactive` and use the `With Actions` pattern below.

```tsx
<Card asChild interactive>
  <button type="button" onClick={handleClick}>
    <Flex direction="col" gap="1">
      <Text className="text-secondary" kind="label/bold/sm">
        DeepSeek
      </Text>
      <Text kind="label/bold/md">DeepSeek-V3.1</Text>
      <Text kind="body/regular/sm">DeepSeek V3.1 Instruct is a hybrid AI model with fast reasoning, 128K context, and strong tool use.</Text>
    </Flex>
  </button>
</Card>
```

### With Actions

Use when the user needs to perform actions without navigating to a detail view, or when multiple distinct actions are available per card. The card itself is not a click target in this pattern — do not also enable `interactive`.

```tsx
<Card>
  Card content... Right aligned actions:
  <Flex gap="2" justify="end" wrap="wrap">
    <Button kind="tertiary">Export</Button>
    <Button kind="secondary">Share</Button>
  </Flex>
  Or you can have actions stretch across the card:
  <Flex gap="2" justify="end" wrap="wrap">
    <Button kind="tertiary" style={{ flex: 1 }}>
      Export
    </Button>
    <Button kind="secondary" style={{ flex: 1 }}>
      Share
    </Button>
  </Flex>
  Or for just a single action:
  <Button color="brand" style={{ width: "100%" }}>
    Share
  </Button>
</Card>
```

### Cards In Responsive Grid

It's common to use cards in a responsive grid. Using the grid component, set a minimum column width and let the cards automatically resize to fill the available space.

```tsx
<Grid colMinWidth="250px" gap="2">
  <Card>Card 1</Card>
  <Card>Card 2</Card>
  <Card>Card 3</Card>
</Grid>
```

### Composed

```tsx
<Flex direction="col" gap="1">
  <CardRoot interactive={false} kind="solid">
    <CardMedia mediaTheme="light" slotHeader={<Badge>Featured</Badge>}>
      <MediaImg />
    </CardMedia>
    <CardContent>Composed card with media</CardContent>
  </CardRoot>
  <CardRoot>
    <CardContent>
      <div className="nv-card-content-header">
        <Badge>Featured</Badge>
      </div>
      Composed card without media
    </CardContent>
  </CardRoot>
</Flex>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ CardContent?: NativeElementAttributes<"div", typeof CardContent>; CardMedia?: NativeElementAttributes<"div", typeof CardMedia> }` | - | Native HTML attributes forwarded to the internal composed components. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces. |
| interactive | `boolean` | `false` | Adds hover and focus affordances to signal the card is clickable. Do not enable on non-clickable cards — it creates false affordance. |
| kind | `"solid" \| "gradient" \| "float"` | `"solid"` | Visual treatment of the card. - "solid" - General-purpose treatment with padded content and a hard border between media and content. Use for most cards. - "gradient" - Same as "solid" but the media fades into the content. Use for promotional or hero-style cards. - "float" - Pairs with `slotMedia` to render the media as a bordered tile while the content sits on the page background without padding. Use when the card should visually lift off the surface. |
| layout | `"horizontal" \| "vertical"` | `"vertical"` | Orientation of the media and content. - "vertical" - Media stacks above the content. Use when the media is the focal point and cards display in a grid. - "horizontal" - Media sits to the left of the content. Use when the content outweighs the media and cards stack in a list. |
| mediaTheme | `"light" \| "dark"` | - | Forces the theme of overlaid `slotHeader` content so it contrasts with the media beneath. - "dark" - For light or bright media; renders overlaid content in the dark theme. - "light" - For dark media; renders overlaid content in the light theme. |
| selected | `boolean` | `false` | Applies the selected visual state, e.g. when the card represents the current choice in a list. |
| slotHeader | `ReactNode` | - | Header content rendered above the body. When paired with `slotMedia`, it overlays the media instead. |
| slotMedia | `ReactNode` | - | Media content rendered above the body, typically an image or video. Pair with `mediaTheme` when also using `slotHeader` so overlaid content remains readable. |

## Exports

- Components: `Card`, `CardContent`, `CardMedia`, `CardRoot`
- Types: `CardContentProps`, `CardMediaProps`, `CardProps`, `CardRootProps`

## See also

- [Panel](./Panel.md)
- [Grid](./Grid.md)
