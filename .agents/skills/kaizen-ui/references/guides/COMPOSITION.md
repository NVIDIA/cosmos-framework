# Composition with `asChild`

Most KUI components accept an `asChild` boolean prop. When set, the component renders its single child element instead of its own DOM node, merging KUI styling, refs, and props onto it. Use it whenever you need a KUI component's behavior on a non-default element.

**Rules:**

- Pass **exactly one** child element.
- Child props take precedence on conflict (so a `className` on the child overrides KUI's class — usually you want to merge with `cn()` if you have access to it).
- The component's accessibility, refs, and event handlers all forward to the child.

**Examples:**

A button that navigates instead of triggering an action:

```tsx
<Button asChild kind="tertiary">
  <a href="/docs">Docs</a>
</Button>
```

A card that wraps a router link:

```tsx
<Card asChild>
  <Link to={`/jobs/${job.id}`}>
    <Card.Header>{job.name}</Card.Header>
    <Card.Body>...</Card.Body>
  </Link>
</Card>
```
