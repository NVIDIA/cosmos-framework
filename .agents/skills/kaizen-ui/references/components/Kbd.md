# Kbd

A visual representation of a keyboard key or shortcut. Place it beside the control or instruction it describes.

## Notes

- Put only key labels inside `Kbd`; keep instructional prose outside it.

- Use `items` for separate keycaps. Pass the full shortcut as children when one compact keycap reads better.

- Match `size` to the text around the shortcut: `small` inside menu items, table cells, and Buttons; `medium` in body copy; `large` in a hero or an empty state.

- Use `kind="flat"` and one joined keycap on colored surfaces. Keep the default `solid` treatment and separate keycaps on neutral surfaces.

- Use `KBD_MAC_KEYS` or `KBD_PC_KEYS` from the same platform signal that controls the keyboard handler and `aria-keyshortcuts`.

- `Kbd` is presentational and does not implement keyboard behavior. Put `aria-keyshortcuts` on the control the shortcut invokes, using ARIA key names such as `Meta+K` or `Control+K`.

## Examples

### Basic Kbd

One keycap holds one key label, and a lone key renders as a bare `<kbd>` with no wrapper. Keep prose like “press” or “to dismiss” outside the component so the key itself stays scannable.

```tsx
<Kbd>Esc</Kbd>
```

### Key Combination

Pass `items` to render a shortcut as one keycap per key. The keys already sit in a row with a narrow gap, so no joiner is needed unless the shortcut is ambiguous without one.

```tsx
<Kbd items={["Cmd", "K"]} />
```

### Kbd In Prose

Separate caps with `slotSeparator="+"` are the clearest form and the right default for prose; one cap holding the whole shortcut suits a tight row where several chips would be noisy.

```tsx
<Stack gap="density-md">
  <p>
    Use <Kbd items={["Ctrl", "Alt", "Delete"]} slotSeparator="+" /> to save all your work.
  </p>
  <p>
    Use <Kbd>Ctrl+Alt+Delete</Kbd> to save all your work.
  </p>
  <p>
    Use <Kbd>⌘⌥Esc</Kbd> to save all your work.
  </p>
</Stack>
```

### Kbd In Menu

Put the shortcut in a menu row's `slotEnd`: the label takes the free space, so every keycap lands on the same right edge without a width or a margin. Keep `kind` unset — a menu row is a neutral container carrying the page's own foreground, so the keys stay `solid` — and set `size="small"` to sit against the row's text. The keycap is decorative, so name the shortcut for assistive tech with `aria-keyshortcuts` on the item itself, using ARIA key names (`Shift+Control+W`) rather than the displayed glyphs. Rows without a shortcut need no placeholder; they align on the same grid.

```tsx
<Menu
  items={[
    {
      children: "New Tab",
      slotEnd: <Kbd items={["Ctrl", "T"]} size="small" />,
      attributes: { MenuItem: { "aria-keyshortcuts": "Control+T" } },
    },
    {
      children: "New Window",
      slotEnd: <Kbd items={["Ctrl", "N"]} size="small" />,
      attributes: { MenuItem: { "aria-keyshortcuts": "Control+N" } },
    },
    {
      children: "Open File",
      slotEnd: <Kbd items={["Ctrl", "O"]} size="small" />,
      attributes: { MenuItem: { "aria-keyshortcuts": "Control+O" } },
    },
    { kind: "divider" },
    {
      children: "Close Window",
      slotEnd: <Kbd items={["Shift", "Ctrl", "W"]} size="small" />,
      attributes: { MenuItem: { "aria-keyshortcuts": "Shift+Control+W" } },
    },
    {
      children: "Close Tab",
      slotEnd: <Kbd items={["Ctrl", "W"]} size="small" />,
      attributes: { MenuItem: { "aria-keyshortcuts": "Control+W" } },
    },
    {
      children: "Save",
      slotEnd: <Kbd items={["Ctrl", "S"]} size="small" />,
      attributes: { MenuItem: { "aria-keyshortcuts": "Control+S" } },
    },
    { kind: "divider" },
    { children: "Share" },
  ]}
/>
```

### Kbd In Tooltip

Use `kind="flat"` in a tooltip. The tooltip surface is an inverse one — `background-color-component-tooltip` is `gray-800` in both themes against `text-color-accent-white` text — so its foreground is never the page's: a `solid` keycap lands as a near-white blob in light mode and disappears into the surface in dark mode, where its fill and the tooltip share `gray-800`. Join the glyphs into one cap rather than passing `items`, since `flat` has no chip to anchor separate caps. Give an icon-only trigger its own `aria-label` — the tooltip describes the control, it does not name it — and put `aria-keyshortcuts` on that trigger, not on the keycap.

```tsx
<Tooltip
  slotContent={
    <>
      Add files{" "}
      <Kbd kind="flat" size="small">
        Ctrl + K
      </Kbd>
    </>
  }
>
  <Button aria-keyshortcuts="Control+K" aria-label="Add files" kind="tertiary" size="small">
    <Icon name="add" />
  </Button>
</Tooltip>
```

### Kbd Inside Button

`kind="flat"` takes the keycap's fill, border, and text from `currentColor`, so the cap and the enclosing component's own label stay accessible and visually aligned with each other in every theme. Reach for it whenever a keycap sits on a surface whose foreground differs from the page's — a brand, danger, or inverse surface — where the default `solid` keycap, built from the neutral surface tokens, lands as a dark blob. Join a multi-key shortcut into one keycap here rather than passing `items`: with no fill and no border, the gaps between separate caps read as uneven spacing inside the button's own text run. Save `items` for a page background, where each cap has a visible chip to anchor it.

```tsx
<Flex align="center" gap="density-md" wrap="wrap">
  <Button color="brand" kind="primary">
    Search{" "}
    <Kbd kind="flat" size="small">
      ⌘K
    </Kbd>
  </Button>
  <Button color="danger" kind="primary">
    Delete{" "}
    <Kbd kind="flat" size="small">
      Del
    </Kbd>
  </Button>
  <Button kind="secondary">
    Dismiss{" "}
    <Kbd kind="flat" size="small">
      Esc
    </Kbd>
  </Button>
</Flex>
```

### Composed Kbd

Reach for the parts when the shortcut is not a uniform list — one key differing in `size`, or a mix of joiners such as `Ctrl + K then S`. Each joiner is its own `KbdSeparator`, so nothing is interleaved for you. Everything else is better served by `items`.

```tsx
<KbdRoot size="small">
  <KbdKey>Ctrl</KbdKey>
  <KbdSeparator>+</KbdSeparator>
  <KbdKey size="large">K</KbdKey>
</KbdRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| items | `ReactNode[]` | - | Key labels rendered as separate keycaps. Takes precedence over `children`. |
| kind | `"flat" \| "solid"` | `"solid"` | Visual treatment inherited by descendant keys and separators unless overridden. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Visual size inherited by descendant keys and separators unless overridden. |
| slotSeparator | `ReactNode` | - | Content placed between keycaps, such as `+` or `then`. |

## Exports

- Components: `Kbd`, `KbdKey`, `KbdRoot`, `KbdSeparator`
- Constants: `KBD_MAC_KEYS`, `KBD_PC_KEYS`
- Types: `KbdKeyProps`, `KbdProps`, `KbdRootProps`, `KbdSeparatorProps`

## See also

- [CodeSnippet](./CodeSnippet.md)
