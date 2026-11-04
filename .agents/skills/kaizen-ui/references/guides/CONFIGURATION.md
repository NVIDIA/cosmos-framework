# Configuration

## Variables and Theme

The design tokens are supplied as CSS variables, which come from `base.css`. Users may also use `base-external.css` for OSS projects(changes the default font) but the `base.css` file includes:

- a CSS reset(replacement for Tailwind's `preflight.css` - use this instead of that)
- the design tokens, light(default, or `.nv-light` class), and dark theme(via `.nv-dark` or `@media(prefers-color-scheme: dark)`)
- fonts

This file can come from:

- the Kaizen UI CDN: `https://webassets.nvidia.com/kaizen-ui-foundations/<version>/base.css` (or `base-external.css`), linked from HTML or fetched at build time — pin `<version>` to the version of the Kaizen UI package you have installed

## Tailwind

To use KUI with Tailwind, we offer a Tailwind plugin. Install Tailwind v4 and add the KUI Foundations plugin to your global CSS file.

> Note we do **not** use the standard Tailwind import because we want to disable CSS layers and remove redundant CSS.

An example ideal `src/global.css` configuration:

```css
@import url("https://webassets.nvidia.com/kaizen-ui-foundations/<version>/base.css"); /* or base-external.css for OSS projects */
@import "@nvidia/foundations-tailwind-plugin";
/**
 * Make sure you set `<html id="style-root">`
 *
 * This ensures your generated Tailwind utilities have higher specificity
 * and override component styles.
 */
#style-root {
  @tailwind utilities;
}

:root {
  /* ensure we're using correct NVIDIA colors by default. Note that sometimes bg-surface-sunken is also acceptable - see [COLORS.md](../design-tokens/COLORS.md#Surface-and-background) */
  @apply bg-surface-base text-primary;
  scrollbar-width: thin;
  scrollbar-color: var(--nv-scrollbar-color);
}
```

You should then find the root `<html>` element in your app and add `id="style-root"` to it, otherwise your generated Tailwind utilities will not work.

## Icons and brand assets

Icons and logos are not part of Kaizen UI — they come from a separate module that varies by project. Examples here show one import specifier, but resolve it against your own setup: import from the icon module the project already depends on rather than adding whatever an example names.

Icon _names_ are the stable part. [`../../assets/icon-catalog.md`](../../assets/icon-catalog.md) is a catalog of design-system icon names, not the export list of any one package — pick the name that matches the action, then map it to the closest equivalent in your icon library. [`../ICONS.md`](../ICONS.md) covers whether to use an icon at all.
