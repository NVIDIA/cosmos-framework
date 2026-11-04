# Carousel

A horizontal container that cycles through a series of content items.

## Notes

- The required `aria-label` (or `aria-labelledby`) should describe the carousel's content, e.g. aria-label="Featured models".

- Children can be any content — Cards, images, custom markup. The Carousel wraps each direct child automatically in a CarouselItem.

- There are no default arrow controls: use <CarouselControls /> for the standard prev/next buttons around a progress indicator. For advanced control you can also use <CarouselArrowButton /> directly.

- Use `itemsPerView` to size items relative to the viewport(e.g. 1 for single item). Use `itemWidth` for a fixed width per item (e.g. itemWidth="260px").

- GOTCHA: `itemsPerView` and `itemWidth` are mutually exclusive. Both should not be used together.

## Examples

### Basic Carousel

Carousel supports any content type. Here is an example using Cards where it defers sizing to the cards.

```tsx
<Carousel
  aria-label="Featured content"
  slotFooter={
    <Flex justify="center">
      <CarouselControls />
    </Flex>
  }
>
  {Array.from({ length: 6 }, (_, index) => ({
    value: index,
    content: `Item ${index + 1}`,
  })).map((item) => (
    <Card key={item.value}>
      <Text kind="body/bold/xl" asChild>
        <p>{item.content}</p>
      </Text>
    </Card>
  ))}
  {/* we could also do images instead, or anything else: */}
  {images.map((image) => (
    <img key={image.alt} src={image.src} alt={image.alt} width={600} height={400} />
  ))}
</Carousel>
```

### With Title And View All Carousel

Fill slotHeader with a title and a navigation action.

```tsx
<Carousel
  aria-label="Featured models"
  slotHeader={
    <Flex align="center" justify="between">
      <Text kind="label/bold/xl">Featured models</Text>
      <Button kind="tertiary">
        View All
        <ChevronRight />
      </Button>
    </Flex>
  }
  slotFooter={
    <Flex justify="center">
      <CarouselControls />
    </Flex>
  }
>
  {cards}
</Carousel>
```

### Split Controls Carousel

Place controls in the header when vertical space below the items is limited, or split the arrows from the indicator across both slots — here buttons in the header with the indicator centered below the items.

```tsx
<Carousel
  aria-label="Featured content"
  slotHeader={
    <Flex align="center" justify="between">
      <Text kind="label/bold/xl">Title</Text>
      <CarouselControls indicator="none" />
    </Flex>
  }
  slotFooter={
    <Flex justify="center">
      <CarouselControls buttonKind="none" />
    </Flex>
  }
>
  {cards}
</Carousel>
```

### Count Indicator Carousel

Use indicator="count" to show the visible item range (e.g. "1-4 of 12") instead of progress bars.

```tsx
<Carousel
  aria-label="Featured content"
  slotFooter={
    <Flex justify="center">
      <CarouselControls buttonKind="tertiary" indicator="count" />
    </Flex>
  }
>
  {cards}
</Carousel>
```

### Items Per View Carousel

Items are content-sized by default. Pass itemsPerView to fit a set number of items in the viewport — fractional values peek the next item. Use itemWidth instead for a fixed width per slide regardless of container size.

```tsx
<Carousel
  aria-label="Featured content"
  itemsPerView={3}
  slotFooter={
    <Flex justify="center">
      <CarouselControls />
    </Flex>
  }
>
  {cards}
</Carousel>
```

### Composed

```tsx
<CarouselRoot aria-label="Featured content">
  <CarouselItems>
    <CarouselItem aria-label="First slide">Slide content</CarouselItem>
    <CarouselItem>Slide content</CarouselItem>
    <CarouselItem>Slide content</CarouselItem>
  </CarouselItems>
  <CarouselControls buttonKind="secondary" indicator="visual" />
</CarouselRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ CarouselItems?: NativeElementAttributes<"div", typeof CarouselItems> }` | - | Native HTML attributes forwarded to the internal composed components. |
| autoplay | `boolean` | `false` | Enables automatic page advancement. |
| autoplayInterval | `number` | `5000` | Time between automatic page advances in milliseconds (5000 ms = 5s). |
| itemsPerView | `number` | - | How many items fit in the viewport. Fractional values peek the next item — i.e. `3.5` shows three and a half, or `1` for single item. Overridden by `itemWidth`. |
| itemWidth | `string \| number` | - | Fixed width of every item. Accepts any CSS length, or numbers which are interpreted as pixels. Overrides `itemsPerView`. |
| loop | `boolean` | `false` | Whether next/previous navigation wraps around at the ends: next on the last page returns to the first, previous on the first jumps to the last, and the arrow buttons never disable. The wrap scrolls across the intermediate slides. |
| onPageChange | `(page: number, meta: { firstVisibleIndex: number; lastVisibleIndex: number; itemCount: number; }) => void` | - | Called when the visible page changes, from scrolling or the controls. |
| renderItem | `false \| ((child: ReactNode, index: number) => ReactNode)` | - | Renders each direct child as a slide. The default wraps each child in a {@link CarouselItem }, leaving children that already are one untouched. Pass `null` to disable the wrapping and render children as-is. |
| slotFooter | `ReactNode` | - | Row below the items, typically a {@link CarouselControls }. The row is unstyled — bring your own layout, e.g. a `Flex` with `justify="center"`. Nothing renders when omitted. |
| slotHeader | `ReactNode` | - | Row above the items, typically a title with an action or a {@link CarouselControls }. The row is unstyled — bring your own layout, e.g. a `Flex` with `justify="between"`. Nothing renders when omitted. |

## Exports

- Components: `Carousel`, `CarouselArrowButton`, `CarouselControls`, `CarouselIndicator`, `CarouselItem`, `CarouselItems`, `CarouselRoot`
- Hooks: `useCarouselContext`
- Types: `CarouselArrowButtonProps`, `CarouselContextValue`, `CarouselControlsProps`, `CarouselIndicatorProps`, `CarouselItemProps`, `CarouselItemsProps`, `CarouselProps`, `CarouselRootProps`, `CarouselRootPropsBase`
