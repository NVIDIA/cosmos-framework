# Resizable

A layout of panels separated by draggable edges, such as a sidebar beside main content or an editor above a terminal.

## Notes

- Give one panel no `defaultSize` so it fills the remaining space; size the others with a percentage number or a CSS length string such as `"264px"`.

- Give every panel, including the one that fills, a `minSize`. When the minimums no longer fit side by side, as at high zoom, the group stacks its panels instead of scrolling sideways.

- Label every panel with `aria-label` or `aria-labelledby`. Each separator takes its accessible name from the panel it resizes, so pass `aria-label` to `ResizableSeparator` only when its panel has no label of its own.

- In composed layouts, render landmarks through `ResizablePanel` with `asChild`, for example a `<nav>` sidebar.

- Persist a layout from `onValueCommit`, and each panel's collapsed state from `onCollapsedChange`, then pass them back through `defaultValue` and `defaultCollapsed`; read them on the server, from a cookie for example, so the first render matches.

- Use `Divider` for a static rule between content, and `SidePanel` for a temporary overlay panel.

## Examples

### Basic Resizable

Give one panel a percentage `defaultSize` and leave the other unsized so it fills the rest. Each separator is named after the panel it resizes.

```tsx
<Resizable
  className="rounded-lg border border-base bg-surface-base"
  items={[
    {
      "aria-label": "Training run",
      children: (
        <div className="p-6">
          <h3 className="mb-3 text-label-bold-lg">Training run</h3>
          <p className="text-secondary">Image classifier · Run 042</p>
          <p className="mt-4">Epoch 8 of 20</p>
        </div>
      ),
      defaultSize: 40,
      minSize: 20,
      value: "preview",
    },
    {
      "aria-label": "Details",
      children: (
        <div className="p-6">
          <h3 className="mb-3 text-label-bold-lg">Run details</h3>
          <p>4 GPUs · Batch size 64</p>
          <p className="mt-4 text-secondary">Resize the panels to make room for the information you need.</p>
        </div>
      ),
      minSize: 20,
      value: "details",
    },
  ]}
  style={{ height: 320 }}
/>
```

### Sidebar Resizable

Size a sidebar in a CSS length so it keeps its width as the window resizes. `collapsible` lets a drag below half of `minSize`, or Enter on the separator, collapse it.

```tsx
<Resizable
  className="rounded-lg border border-base bg-surface-base"
  items={[
    {
      "aria-label": "Documentation",
      children: (
        <VerticalNav
          style={{ width: "100%", borderRightWidth: 0 }}
          items={[
            {
              id: "intro",
              active: true,
              children: "Introduction",
              href: "/intro",
            },
            { id: "install", children: "Installation", href: "/install" },
            { id: "metrics", children: "Metrics", href: "/metrics" },
          ]}
        />
      ),
      collapsible: true,
      defaultSize: "264px",
      maxSize: "400px",
      minSize: "200px",
      value: "sidebar",
    },
    {
      "aria-label": "Page",
      children: (
        <article className="p-6">
          <h3 className="mb-3 text-label-bold-lg">Introduction</h3>
          <p>Build and monitor GPU workloads from one workspace.</p>
          <p className="mt-4 text-secondary">Keep navigation nearby, or collapse it to focus on your work.</p>
        </article>
      ),
      minSize: "240px",
      value: "page",
    },
  ]}
  style={{ height: 320 }}
/>
```

### Vertical Resizable

Stack panels with `orientation="vertical"`, as for an editor above a terminal.

```tsx
<Resizable
  className="rounded-lg border border-base bg-surface-base"
  items={[
    {
      "aria-label": "Editor",
      children: (
        <div className="p-6">
          <h3 className="mb-3 text-label-bold-lg">training.yaml</h3>
          <pre className="text-sm">{"model: image-classifier\nbatch_size: 64\nepochs: 20"}</pre>
        </div>
      ),
      defaultSize: 60,
      minSize: 30,
    },
    {
      "aria-label": "Terminal",
      children: (
        <div className="bg-surface-raised p-6">
          <h3 className="mb-3 text-label-bold-lg">Training logs</h3>
          <pre className="text-sm">{"[09:41:02] Dataset loaded\n[09:41:03] Training started"}</pre>
        </div>
      ),
      collapsible: true,
      minSize: "120px",
    },
  ]}
  orientation="vertical"
  style={{ height: 400 }}
/>
```

### Controlled Collapse Resizable

Control `collapsed` to drive a panel from a toolbar button. `onCollapsedChange` keeps the button in step when someone collapses the panel from its separator.

```tsx
() => {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <Flex direction="col" gap="density-md">
      <Button kind="tertiary" onClick={() => setCollapsed(!collapsed)}>
        {collapsed ? "Show sidebar" : "Hide sidebar"}
      </Button>
      <Resizable
        className="rounded-lg border border-base bg-surface-base"
        items={[
          {
            "aria-label": "Sidebar",
            children: (
              <div className="p-6">
                Run 042
                <br />
                Image classifier
              </div>
            ),
            collapsed,
            collapsible: true,
            defaultSize: "240px",
            minSize: "160px",
            onCollapsedChange: setCollapsed,
          },
          {
            "aria-label": "Page",
            children: <div className="p-6">Training workspace</div>,
            minSize: "240px",
          },
        ]}
        style={{ height: 240 }}
      />
    </Flex>
  );
};
```

### Composed

Nest a root inside a panel for two-way layouts. Where two separators meet, dragging moves both.

```tsx
<ResizableRoot className="rounded-lg border border-base bg-surface-base" style={{ height: 400 }}>
  <ResizablePanel aria-label="Files" defaultSize="240px" minSize="160px">
    <div className="p-6">
      <h3 className="mb-3 text-label-bold-lg">Files</h3>
      <p>training.yaml</p>
      <p className="mt-2 text-secondary">dataset.csv</p>
    </div>
  </ResizablePanel>
  <ResizableSeparator />
  <ResizablePanel aria-label="Workspace" minSize="240px">
    <ResizableRoot orientation="vertical">
      <ResizablePanel aria-label="Editor" defaultSize={60} minSize={30}>
        <div className="p-6">
          <h3 className="mb-3 text-label-bold-lg">training.yaml</h3>
          <pre className="text-sm">{"model: image-classifier\nbatch_size: 64\nepochs: 20"}</pre>
        </div>
      </ResizablePanel>
      <ResizableSeparator />
      <ResizablePanel aria-label="Terminal" minSize="120px">
        <div className="p-6">
          <h3 className="mb-3 text-label-bold-lg">Training logs</h3>
          <p className="text-secondary">Epoch 8 of 20 · Loss 0.24</p>
        </div>
      </ResizablePanel>
    </ResizableRoot>
  </ResizablePanel>
</ResizableRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `ResizablePanelProps[]` | - | Panels in order. A separator is placed between each pair. |
| attributes | `{ ResizableSeparator?: NativeElementAttributes<"hr", typeof ResizableSeparator> }` | - | Native HTML attributes forwarded to the internal composed components. They apply to every separator, so label separators one by one with the composed parts. |
| defaultValue | `{ [x: string]: ResizableSize; }` | - | Panel sizes to start from when uncontrolled, such as a layout restored from a cookie. Panel `defaultSize` still decides where a reset returns to. |
| onValueChange | `(value: { [x: string]: ResizableSize; }) => void` | - | Called on every size change while a separator moves, with each sized panel's size in the unit it was authored in. |
| onValueCommit | `(value: { [x: string]: ResizableSize; }) => void` | - | Called once a resize finishes, on pointer release, a keyboard step, or a placement, with the committed sizes. Persist layouts from here. |
| orientation | `"horizontal" \| "vertical"` | `"horizontal"` | Whether panels sit side by side or stacked, and so which way separators move. |
| showHandle | `boolean` | `true` | Shows the grip on every separator. Without it each edge widens to a 24px target and its line gains contrast. |
| value | `{ [x: string]: ResizableSize; }` | - | Controlled panel sizes keyed by panel `value`. Pair with `onValueChange` to manage state externally. |

`* = required prop`

## Exports

- Components: `Resizable`, `ResizablePanel`, `ResizableRoot`, `ResizableSeparator`
- Types: `ResizableLayout`, `ResizableOrientation`, `ResizablePanelProps`, `ResizableProps`, `ResizableRootProps`, `ResizableSeparatorProps`, `ResizableSize`

## See also

- [Divider](./Divider.md)
- [SidePanel](./SidePanel.md)
