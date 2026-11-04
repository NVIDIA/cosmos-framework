# Avatar

A small visual identifier for a user, account, or entity. Prioritizes an image when available and falls back to initials or an icon otherwise.

## Notes

- Always provide a meaningful `fallback` (typically the user's initials) — never leave it empty, since the image may fail to load.

- Provide descriptive `alt` text whenever an `src` is set. Use `alt=""` only when the user's name is already displayed next to the avatar to avoid screen-reader redundancy.

- Set `interactive` only when clicking the avatar actually navigates or triggers an action. The hover state otherwise implies an affordance that does not exist.

- Pick `size` to match the context's density: `"small"` for lists and tables, `"medium"` for standard placements, `"xlarge"` or `"xxlarge"` for profile or hero pages.

## Examples

### Basic Avatar

Always set `fallback` (typically initials) so the avatar still renders if the image fails to load. Pass descriptive `alt` text identifying the user or entity.

```tsx
<Avatar fallback="AM" alt="Alex Morgan" src="users-am.jpg" size="xlarge" />
```

### With Name

Set `alt=""` when the user's name is already shown next to the avatar — screen readers would otherwise announce the name twice.

```tsx
<div className="flex items-center gap-2">
  <Avatar
    fallback="AM"
    // use empty alt to avoid redundancy with name text
    alt=""
    src="users-am.jpg"
  />
  <Text kind="label/bold/sm">Alex Morgan</Text>
</div>
```

### Interactive Avatar

Sometimes Avatars are used as triggers - in that case use the `interactive` prop so the Avatar has hover states.

```tsx
<Avatar fallback="AM" alt="Alex Morgan" src="users-am.jpg" interactive />
```

### Icon Avatar

To render an Avatar with an icon, use the fallback prop with an icon component.

```tsx
<Avatar fallback={<Cog />} />
```

### As Link Or Button

To render an Avatar as a link or button, use the asChild prop on AvatarRoot to render it as its child element.

```tsx
<AvatarRoot asChild interactive size="large">
  <a
    // or use a button element, depending on the use case
    href="https://example.com"
    rel="noreferrer noopener"
    target="_blank"
  >
    <AvatarImage alt="Alex Morgan" src="users-am.jpg" />
    <AvatarFallback>AM</AvatarFallback>
  </a>
</AvatarRoot>
```

### Composed

```tsx
<AvatarRoot size="xxlarge">
  <AvatarImage alt="Alex Morgan" src="users-am.jpg" />
  <AvatarFallback>AM</AvatarFallback>
</AvatarRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **fallback** * | `ReactNode` | - | Content rendered when no image source is provided or the image fails to load; typically the user's initials or an icon. |
| alt | `string` | - | Accessible label for the image; prefer the user's full name. Set to an empty string when the name is already displayed nearby to avoid screen reader redundancy. |
| attributes | `{ AvatarImage?: NativeElementAttributes<"img", typeof AvatarImage>; AvatarFallback?: NativeElementAttributes<"div", typeof AvatarFallback> }` | - | Native HTML attributes forwarded to the internal composed components. |
| interactive | `boolean` | - | Enables hover, active, and pointer styling on the avatar. Defaults to `true` when `onClick` is provided, `false` otherwise. |
| kind | `"outline" \| "solid"` | `"outline"` | The visual treatment of the avatar. - "outline" - Adds a border ring; preferred for dense layouts. - "solid" - No outline. |
| size | `"small" \| "medium" \| "large" \| "xlarge" \| "xxlarge"` | `"medium"` | The size of the avatar. - "small" - Lists, tables, and other dense rows. - "medium" - General-purpose default for most layouts. - "large" - Profile detail pages and prominent placements. - "xlarge" - Featured profile or settings headers. - "xxlarge" - Hero or banner placements. |
| src | `string` | - | URL of the image to display. |

`* = required prop`

## Exports

- Components: `Avatar`, `AvatarFallback`, `AvatarImage`, `AvatarRoot`
- Types: `AvatarFallbackProps`, `AvatarImageProps`, `AvatarProps`, `AvatarRootProps`
