# PortalProvider

Sets default portal behavior for descendant components that support portaling.
Wrap the app (or a region) to make overlays portal to a shared target without
passing `portal`/`portalContainer` to each one; descendants still override with
their own props.

## Notes

- Render a single `PortalProvider` near the app root to make modals,
  popovers, tooltips, and other overlays portal by default. This is the dedicated
  home for app-wide portal defaults — `ThemeProvider` only controls theming. Set
  `ssrStrategy="defer"` to render those overlays only after hydration, avoiding a
  flash of unportaled content on the server.

## Examples

### App Root Portal Provider

Wrap your app in a `PortalProvider` so descendant overlays (modals, popovers, tooltips) portal to the document body by default, escaping ancestor styles. Individual components opt out with `portal={false}`.

```tsx
<PortalProvider>
  <Panel>App content</Panel>
</PortalProvider>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| portal | `boolean` | `true` | Default for whether descendant components portal their floating content. Components override this with their own `portal` prop. |
| portalContainer | `HTMLElement` | `document.body` | Default container for descendant portaled content when no nearer portal target exists. |
| ssrStrategy | `"inline" \| "defer"` | `"inline"` | Server-rendering strategy for descendant portals. - "inline" - Renders content in place on the server so it remains available without JavaScript. - "defer" - Omits portaled content until hydration to avoid a flash of unportaled content. |

## Exports

- Components: `PortalProvider`
- Types: `PortalProviderProps`
