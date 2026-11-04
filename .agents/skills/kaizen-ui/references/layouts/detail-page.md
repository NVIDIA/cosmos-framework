# Detail Page (Layout Reference)

For pages that display an entity's current state/information.

## When to use

Refer to [detail-placement.md](./detail-placement.md) to determine if you need a standalone detail page, a side panel, or both.

In general, use a detail page when:

- URL represents a single entity (e.g., `/{entity-type}/{id}`)
- Primary user intent is understanding current state
- Content is read-only or mostly read-only

## Content Layout

Once you've decided the entity warrants a standalone page, the next decision is _how to lay out the page body_. This is a **content-driven decision**, not a template choice.

Detail pages use the 80rem (1280px) maximum width from the shared page shell.

### Layout signals

| Signal                        | Favors multi-column                                                                                        | Favors single-column / stacked grid                                      |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Content types on the page** | Heterogeneous — primary editable data + metadata + status + summaries belong in spatially distinct regions | Homogeneous — all metadata, all identity, or all similar-weight sections |
| **Content volume**            | Enough content to fill both columns without forced emptiness                                               | Sparse (~10 or fewer flat fields) or one section clearly dominates       |
| **Viewport use**              | Content would otherwise leave significant whitespace at ≥1024px                                            | Content fits comfortably stacked without awkward gaps                    |
| **Scan pattern**              | Parallel scanning aids comprehension (user compares primary to metadata)                                   | Sequential / single-path reading                                         |

**When the signals are mixed, default to stacked with a responsive grid.** Forcing columns onto ambiguous content is a more visible failure than stacking content that could have worked in columns.

### Two valid layout patterns

#### Pattern A — Stacked / responsive grid (default for KV-heavy bodies)

Use when content is homogeneous (mostly identity + light state), the entity is sparse, or one section clearly dominates.

Single column with a responsive KV grid. Group content by section with headings.

#### Pattern B — Two-column (primary / secondary)

Use when there's a clearly dominant "primary" section (core function, active workload, heavy data) and a distinct secondary section (metadata, configuration, reference). Content types are **heterogeneous** — typically a chart, diagram, transcript, or workflow on the primary side and a metadata/config sidebar on the secondary side.

**Check content balance before committing.** Pattern B assumes the primary column holds the _heaviest_ content. If, after sketching the page, the right (secondary) column accumulates more sections than the left, the assumption is wrong — either rebalance the content map, or use Pattern A.

### Tabbed views

Use tabs when the page has 2+ distinct content categories that each warrant their own view and require different mental models. Place the tab pattern above the primary content.

**Containment of tab content.** Tabs do **not** add containment. Content inside each tab adopts whatever containment its content type normally uses:

- A KV grid → no container, headings + spacing only
- A table → renders directly inside the tab area; do **not** add a redundant container
- A chart widget → use one labeled visual container

## Status surfacing — default + fallback patterns

Detail pages have two distinct status surfaces, and they answer different questions:

| Surface                                                     | What it answers                                                                                  | Pattern                                   |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| **Entity's primary status** ("is this thing OK?")           | The single semantic state — Healthy, Degraded, Pending, Active, Failed                           | **Badge inline with the page heading**    |
| **Live operational metrics** ("what's it doing right now?") | Short-lived, frequently-changing measurements — utilization, allocation, freshness, last-updated | **Mini-tile grid** below the page heading |

**Default detail-page status surface = heading Badge + mini-tile grid.** Use both unless the entity has neither a meaningful primary status nor operational metrics to surface.

### Why both, not one or the other

The primary status is _categorical and semantic_. It belongs with the heading because it qualifies the entity name (the user reads "dfw-hgx-h100-prod, **Degraded**" as a single phrase about identity). The mini-tile grid is _quantitative and operational_ — the metrics the user scans before acting.

### Mini-tile composition (required when surface is used)

Each mini-tile uses a 3-line composition: metric label / value (with optional denominator) / supporting context. The supporting context line is **required** — it tells the user how to interpret the value (temporal scope, denominators, salient identifiers).

See [data-display.md](../patterns/data-display.md) for tile typography. Detail-page-specific composition: arrange tiles in a responsive CSS Grid row that starts at one column and adds columns as space permits. For five tiles, target two columns around 36rem, three around 48rem, and five around 62rem while keeping every tile readable.

### Sparse-entity fallback

If the entity has neither a meaningful primary status nor enough metrics to fill 3+ tiles, fall back to **inline KV fields** within the body — render `Status` as one field among several.

---

## Regions — composable building blocks

### Identity (mandatory)

Use [PageHeader](../components/PageHeader.md) for identity. Place breadcrumbs with the heading; never use a standalone Back button.

Detail pages use `PageHeader kind="flat"` and place breadcrumbs in `slotBreadcrumbs`.

**Primary action styling:** use the [Button](../components/Button.md) brand treatment for positive actions such as Edit, Deploy, Save, and Run. Use the danger treatment plus the [confirmation pattern](../patterns/confirmations.md) for destructive actions.

---

## Content hierarchy heuristic

Assign content by asking: **What question does this field answer?**

| Question                 | Where it goes                                                                         | Prominence                                        |
| ------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Is it working?           | Status-surfacing pattern (badge in header / mini-tile grid / inline KV badge)         | Visually distinct via `Badge` with semantic color |
| What does it do?         | Primary section (Pattern A top section or Pattern B primary column)                   | Most space                                        |
| Where does it run?       | Secondary section (Pattern A sibling section or Pattern B secondary column)           | Supporting context                                |
| What limits apply?       | Secondary section                                                                     | Reference info                                    |
| Details I rarely need?   | Collapsed section                                                                     | Revealed on demand                                |
| What happened over time? | Separate tab. Refer to [activity-feed.md](../patterns/activity-feed.md) for guidance. | Logs, history, events                             |

## React examples

### Stacked detail

Default detail-page body for KV-heavy entities. Outer shell is owned by layout-principles.md.

```tsx
<Stack gap="6">
  {/* Page shell: layout-principles.md (max-w-7xl <main>). Body-only. */}
  <PageHeader
    kind="flat"
    slotBreadcrumbs={<Breadcrumbs items={breadcrumbItems} />}
    slotHeading={
      <Flex gap="density-md" align="center">
        <h1>dfw-hgx-h100-prod</h1>
        <Badge kind="solid" color="yellow">
          Degraded
        </Badge>
      </Flex>
    }
    slotDescription={<span className="text-secondary">Production H100 inference cluster — DFW region.</span>}
    slotActions={
      <>
        <Button kind="tertiary">Edit</Button>
        <Button color="brand">Deploy</Button>
      </>
    }
  />
  <Grid gap="6" colMinWidth="170px">
    {tiles.map((tile) => (
      <Panel className="p-density-xl" key={tile.label}>
        <Stack gap="3">
          <Text kind="body/regular/sm">{tile.label}</Text>
          <Text kind="label/semibold/2xl">{tile.value}</Text>
          <Text kind="body/regular/sm" className="text-secondary">
            {tile.context}
          </Text>
        </Stack>
      </Panel>
    ))}
  </Grid>
  <Tabs
    defaultValue="overview"
    items={[
      { value: "overview", children: "Overview" },
      { value: "events", children: "Events" },
    ]}
  />
  <Stack gap="8">
    <Stack gap="4">
      <Text kind="body/bold/lg">Cluster information</Text>
      <Grid gap="6" colMinWidth="300px">
        {clusterKv.map((kv) => (
          <Stack key={kv.label} gap="1">
            <Text kind="label/regular/sm" className="text-secondary">
              {kv.label}
            </Text>
            <Text kind="body/semibold/md">{kv.value}</Text>
          </Stack>
        ))}
      </Grid>
    </Stack>
    <Stack gap="4">
      <Text kind="body/bold/lg">Scheduling &amp; lifecycle</Text>
      <Grid gap="6" colMinWidth="300px">
        {schedulingKv.map((kv) => (
          <Stack key={kv.label} gap="1">
            <Text kind="label/regular/sm" className="text-secondary">
              {kv.label}
            </Text>
            <Text kind="body/semibold/md">{kv.value}</Text>
          </Stack>
        ))}
      </Grid>
    </Stack>
  </Stack>
</Stack>
```

### Two-column detail

```tsx
<Stack gap="6">
  <PageHeader
    kind="flat"
    slotBreadcrumbs={<Breadcrumbs items={breadcrumbItems} />}
    slotHeading={
      <Flex gap="3" align="center">
        <h1>dfw-hgx-h100-prod</h1>
        <Badge kind="solid" color="yellow">
          Degraded
        </Badge>
      </Flex>
    }
    slotActions={
      <>
        <Button kind="tertiary">Edit</Button>
        <Button color="brand">Deploy</Button>
      </>
    }
  />
  <Tabs defaultValue="overview" items={tabsItems} />
  <Grid className="grid-cols-1 justify-center lg:grid-cols-[minmax(0,954px)_302px]" gap="6">
    <Stack gap="8">
      <Stack gap="3">
        <Text kind="body/bold/lg">Running PyTorch Using Docker</Text>
        <Text kind="body/regular/md">Use this container when the workflow needs a supported PyTorch runtime with CUDA, NCCL, and common data-science libraries already configured.</Text>
      </Stack>
      <Stack gap="3">
        <Text kind="body/bold/lg">Before you start</Text>
        <Text kind="body/regular/md">
          Verify Docker, NVIDIA Container Toolkit, and driver compatibility before running GPU workloads. Keep setup guidance in the primary column because it is part of the user's reading flow.
        </Text>
      </Stack>
      <Stack gap="3">
        <Text kind="body/bold/lg">Example command</Text>
        <CodeSnippet language="bash" value={dockerCommand} copyButton="inline" />
      </Stack>
    </Stack>

    <aside className="w-full">
      <Stack gap="6">
        <Stack gap="2">
          <Text kind="body/bold/md">Metadata</Text>
          <KeyValueGroup items={metadataKv} />
        </Stack>
        <Stack gap="2">
          <Text kind="body/bold/md">Tags</Text>
          <Flex gap="2" wrap="wrap">
            {detailTags.map((tag) => (
              <Tag key={tag} color="blue" kind="outline" readOnly>
                {tag}
              </Tag>
            ))}
          </Flex>
        </Stack>
      </Stack>
    </aside>
  </Grid>
</Stack>
```
