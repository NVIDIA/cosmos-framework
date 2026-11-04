# ThemeProvider

Establishes a theme context for descendant components, controlling color theme and density. Nest providers to scope different settings to specific parts of the app.

## Notes

- For app-wide theming, set `global` on the outermost provider so portaled components inherit the theme too.

- This is an optional component. Apps may choose to just manage light/dark mode via CSS classes manually, and that's fine

- Avoid hardcoding `theme="dark"` or `theme="light"` at the root — it ignores the user's OS preference and violates accessibility expectations. Prefer `theme="system"` and let users override via app settings.

- Scope density changes to layout regions by nesting a ThemeProvider, not per individual component. Common pattern: `density="compact"` around a data table or SidePanel while the rest of the page stays `"standard"`.

- Use `density="spacious"` sparingly — reserve it for marketing or onboarding surfaces. `"standard"` and `"compact"` cover the rest.

## Examples

### Basic Theme Provider

```tsx
<ThemeProvider density="compact">Density-aware children</ThemeProvider>
```

### App Root Theme Provider

Mount once at the app root with `global` so portaled components (modals, popovers, tooltips) inherit the theme. Default `theme="system"` respects the user's OS preference.

```tsx
<ThemeProvider theme="system" density="standard" global>
  <Panel>App content</Panel>
</ThemeProvider>
```

### Scoped Density Theme Provider

Nest a ThemeProvider to override density for a specific region (data tables, side panels) without affecting the rest of the page.

```tsx
<ThemeProvider density="standard">
  <ThemeProvider density="compact">
    <Panel>Compact region (e.g. SidePanel, data table)</Panel>
  </ThemeProvider>
</ThemeProvider>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| asChild | `boolean` | - | Render-as-child slot (from Radix Primitive) |
| defer | `boolean` | `false` | Defers theme class application until after hydration. Use when integrating with another theme solution such as `next-themes`. `suppressHydrationWarning` is applied automatically. |
| density | `"compact" \| "standard" \| "spacious"` | `"standard"` | Spacing density applied to descendant components. - "compact" - Minimal spacing for information-dense layouts such as data tables and admin panels. - "standard" - Balanced spacing for general-purpose interfaces. - "spacious" - Maximum spacing for touch-first and accessibility-focused interfaces. |
| global | `boolean` | `false` | Applies theme classes to the document `target` element so portals and overlays are also themed. Use on the outermost provider for app-wide theming. |
| motion | `"disabled" \| "system"` | `"system"` | Motion preference applied to component animations. - "disabled" - Disables all built-in animations and transitions. - "system" - Follows the user's `prefers-reduced-motion` preference. |
| target | `"body" \| "html"` | `"html"` | The document element to receive theme classes when `global` is true. |
| theme | `"light" \| "dark" \| "system"` | - | Color theme applied to descendant components. - "light" - Forces the light theme regardless of system preference. - "dark" - Forces the dark theme regardless of system preference. - "system" - Follows the user's `prefers-color-scheme` preference. |

## Exports

- Components: `ThemeProvider`
- Hooks: `useParentThemeContext`, `useTheme`, `useThemeContext`, `useThemeManager`
- Contexts: `ThemeContext`
- Utilities: `createReducedMotionListener`, `createSystemThemeListener`, `detectReducedMotionPreference`, `detectSystemThemePreference`, `getDensityClassName`, `getThemeClassName`, `resolveMotion`, `resolveTheme`
- Constants: `DENSITY_CLASS_NAMES`, `MOTION_CLASS_NAMES`, `PREFERS_COLOR_SCHEME_DARK_QUERY`, `PREFERS_REDUCED_MOTION_QUERY`, `THEME_CLASS_NAMES`
- Types: `Density`, `Motion`, `ResolvedTheme`, `Theme`, `ThemeContextType`, `ThemeProviderProps`
