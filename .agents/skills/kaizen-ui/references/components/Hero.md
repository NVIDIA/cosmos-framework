# Hero

A large page banner that combines headline messaging and calls to action with imagery.

## Notes

- Content is left-aligned by default; apply `text-center` or `text-right` on the Hero to change alignment.

- Hero is for landing pages, feature announcements, and onboarding surfaces — not standard application pages. For standard page headers use `PageHeader`.

- Include at least one call-to-action button in `slotActions` — Hero is meant to drive action.

- Keep `slotBody` to 2-3 sentences. Hero is for impact, not long-form reading.

- Constrain Hero with a responsive `min-height` (e.g. `style={{ minHeight: 'clamp(440px, 60vh, 560px)' }}` on a wrapper) so short content does not collapse the band, when you want to grow the hero beyond fitting the content. The 440–560px range keeps the CTA, headline, and subhead reading as one gesture without pushing content below the fold on 1080p screens.

- Padding alone is not enough — without a `min-height`, short content collapses the hero into a narrow band and breaks page rhythm.

## Examples

### Basic Hero

Use for prominent page banners that communicate key messaging without media.

```tsx
<Hero slotHeading="Welcome to the Platform" slotBody="Get started by exploring available models, tools, and resources." />
```

### With Subheading Hero

Use when a secondary label above the heading provides additional context like a category or publisher name.

```tsx
<Hero slotSubheading="Getting Started" slotHeading="Build with NVIDIA" slotBody="Access pre-trained models, SDKs, and APIs to accelerate your AI projects." />
```

### With Actions Hero

Use when the hero needs prominent call-to-action buttons below the body text.

```tsx
<Hero
  slotSubheading="New Release"
  slotHeading="Model Hub"
  slotBody="Discover, customize, and deploy foundation models."
  slotActions={
    <>
      <Button kind="secondary">Learn More</Button>
      <Button color="brand">Get Started</Button>
    </>
  }
/>
```

### With Media Hero

Pass `slotMedia` for a hero with a background image. By default content in the Hero will use the current theme, but to ensure content is readable you can set `mediaTheme` to `dark` or `light` so the content renders with a fixed theme.

```tsx
<Hero mediaTheme="dark" slotHeading="Deep Learning Institute" slotBody="Hands-on training for developers, data scientists, and researchers." slotMedia={<Media />} />
```

### Custom Heading Hero

Use when the heading needs richer content than a string. slotHeading accepts any ReactNode, so you can compose layout primitives, badges, or icons inline.

```tsx
<Hero
  slotSubheading="Kaizen Design System"
  slotHeading={
    <Flex direction="col" gap="density-md">
      <span>Kaizen UI Foundations</span>
      <Flex gap="density-md" wrap="wrap">
        <Badge kind="solid">Accessible</Badge>
        <Badge kind="solid">Responsive</Badge>
        <Badge kind="solid">CSS Only</Badge>
      </Flex>
    </Flex>
  }
  slotBody="A CSS-only implementation of the NVIDIA Design Language."
/>
```

### Aligned Hero

Use the text-center or text-right utility class on the Hero when the design calls for non-default text alignment, and align the footer actions to match. Left alignment is preferred for most cases.

```tsx
<Hero
  attributes={{ HeroFooter: { className: "justify-center" } }}
  className="text-center"
  slotSubheading="Getting Started"
  slotHeading="Centered Hero"
  slotBody="Hero content is left-aligned by default. Apply the text-center or text-right utility class to the Hero to override the alignment."
  slotActions={
    <>
      <Button kind="secondary">Learn More</Button>
      <Button color="brand">Get Started</Button>
    </>
  }
/>
```

### Custom Styling Hero

Use the Hero CSS variables (--padding, --max-width, --text-color) to customize spacing, width, or text color while keeping the rest of the component's styling.

```tsx
<Hero
  style={{ "--padding": "0", "--max-width": "100%" }}
  slotHeading="Edge-to-Edge Hero"
  slotBody="Override the --padding, --max-width, and --text-color CSS variables to fine-tune hero spacing and width without forking the component."
  slotActions={<Button color="brand">Get Started</Button>}
/>
```

### Composed

```tsx
<HeroRoot>
  <HeroContent>
    <HeroSubheading>Subheading</HeroSubheading>
    <HeroHeading>Hero Heading</HeroHeading>
    <HeroBody>Hero body content with a description of the page or feature.</HeroBody>
    <HeroFooter>
      <Button color="brand">Primary Action</Button>
    </HeroFooter>
  </HeroContent>
</HeroRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **slotBody** * | `ReactNode` | - | Supporting body copy rendered between the heading and the actions. |
| **slotHeading** * | `ReactNode` | - | Main heading text. Scales responsively with the component width. |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| attributes | `{ HeroMedia?: NativeElementAttributes<"div", typeof HeroMedia>; HeroContent?: NativeElementAttributes<"div", typeof HeroContent>; HeroHeading?: NativeElementAttributes<"div", typeof HeroHeading>; HeroSubheading?: NativeElementAttributes<"div", typeof HeroSubheading>; HeroBody?: NativeElementAttributes<"div", typeof HeroBody>; HeroFooter?: NativeElementAttributes<"div", typeof HeroFooter> }` | - | Native HTML attributes forwarded to the internal composed components. |
| mediaTheme | `"light" \| "dark"` | - | Tonal context of the background media so foreground text and components flip to the readable theme. Set `"dark"` for dark imagery and `"light"` for light imagery. |
| slotActions | `ReactNode` | - | Call-to-action area rendered at the bottom of the hero, typically containing one or more buttons. |
| slotMedia | `ReactNode` | - | Media rendered as the background of the hero. Pair with `mediaTheme` to keep overlaid content readable. |
| slotSubheading | `ReactNode` | - | Smaller text rendered above the heading. Scales responsively with the component width. |

`* = required prop`

## Exports

- Components: `Hero`, `HeroBody`, `HeroContent`, `HeroFooter`, `HeroHeading`, `HeroMedia`, `HeroRoot`, `HeroSubheading`
- Types: `HeroBodyProps`, `HeroContentProps`, `HeroFooterProps`, `HeroHeadingProps`, `HeroMediaProps`, `HeroProps`, `HeroRootProps`, `HeroSubheadingProps`

## See also

- [PageHeader](./PageHeader.md)
