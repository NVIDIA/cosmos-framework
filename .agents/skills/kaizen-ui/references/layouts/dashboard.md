# Dashboard Page (Layout Reference)

Use for dashboard pages that present aggregated data, metrics, and visualizations across multiple entities — answering "what's happening" at a system or portfolio level rather than "what is this one thing."

## When to use

- The page shows metrics, trends, or status across a collection of entities (not a single entity)
- Primary user intent is monitoring, assessing, or comparing — not editing
- The content includes at least two of: stat blocks, charts, tables, status summaries
- The data has a shared scope (time range, team, cluster, environment) that filters should control globally

**Examples:** cluster utilization overview, operations dashboard, team allocation summary, system health monitor, pipeline throughput.

## When NOT to use a dashboard page

| Situation                                         | Instead use                  |
| ------------------------------------------------- | ---------------------------- |
| Single entity's state (one cluster, one request)  | Detail page                  |
| Browsing/filtering a list to find a specific item | List view                    |
| Editing configuration                             | Form page                    |
| Single chart or metric inline with other content  | Embed in detail or list page |

## See Also

- Read [data display patterns](../patterns/data-display.md) to ensure you are making the right decisions

---

## Information hierarchy (read before building)

### The inverted pyramid

1. **Top layer — Status and targets.** Stat blocks answering "are we on track?" at a glance. First thing the user sees; should answer the biggest questions in under 5 seconds.
2. **Middle layer — Trends and comparisons.** Charts that explain _movement_ — is utilization going up or down, which team is consuming the most, are clusters healthy or degrading.
3. **Bottom layer — Detail and follow-up.** Tables and lists that route the user to specific entities for action.

Each layer is less urgent and more detailed than the one above.

### Natural scanning order

Eye-tracking research (NN/g) shows a strong top-left bias:

- **Place the most critical KPIs in the top-left quadrant** — eyes land there first.
- **Order stat blocks by priority left to right.** Attention decays across a row — the first 3-4 stat blocks get the most attention; the last 1-2 are often barely registered.
- **Left-column charts are scanned before right-column charts** at the same vertical position.

### Progressive disclosure

- Show aggregate metrics first; chart segments link to filtered detail views.
- Keep chart titles and stat-block labels terse — use tooltips for explanation, not body text under every chart.
- Truncate dashboard tables to ~10 rows with "View all" rather than rendering hundreds.

### Actionability

A dashboard that only _displays_ data without enabling action is a wall chart. Every region should connect to the next step:

- Stat blocks with warning/critical values should visually flag the anomaly (color, icon).
- Chart segments should be clickable to navigate to a filtered view of the underlying data.
- Table rows should link to entity detail or actions.

---

## Page shell + structure

Use the standard page shell from [page-shell.md](./page-shell.md).

Note: No decorative backgrounds. The design system uses a flat surface background. Do not add gradients, glows, or color washes — these visually compete with chart and stat content.

---

## Regions

### Page header

Use `PageHeader` with `kind="flat"`. Same rules as list views.

- **`slotHeading`:** dashboard title (e.g., "Cluster Utilization")
- **`slotDescription`:** one line describing scope or data source
- **`slotActions`:** (optional) - Export, settings, view controls. **Refresh does not belong here** — it belongs in the filter toolbar (see below) so it sits next to the data-freshness signal it controls.

### Global filters toolbar

Global filters control the scope of **all** data on the page — time range, team, environment, cluster group. Use a split-layout toolbar: filter inputs on the left, page-level metadata + Refresh on the right. The Figma dashboard toolbar is 40px tall, with three 200px controls at 8px gaps, then timestamp + Refresh aligned right.

**Rules:**

- Time range is almost always a global filter for dashboards. Place it first unless search is the dominant narrowing control for the dataset.
- 1–4 filters: inline in the toolbar row. 5–7 filters: inline with wrap. 8+: a single filters [Dropdown](../components/Dropdown.md) or button opening a [SidePanel](../components/SidePanel.md) when filters need more room.
- All controls use the design system's **default size (medium)**. Do not use `size="small"` on a dashboard.

**Per-chart filters** (zoom, series toggle) live inside the chart Panel, not in the global filter bar.

### System messages (stale data, errors, announcements)

**Position:** between global filters and stat blocks.

When the dashboard needs to communicate a system-level condition — stale metrics, partial data, load errors, maintenance notices — use [Banner](../components/Banner.md) `kind="inline"`.

| Status             | Use for                                                  |
| ------------------ | -------------------------------------------------------- |
| `status="warning"` | Stale data, partial metrics, degraded data source        |
| `status="error"`   | Data load failure, system unavailable                    |
| `status="info"`    | Announcements, maintenance notices, data-freshness notes |

### Stat blocks (KPI row)

3-6 key metrics at the top of the page. See the [example](#dashboard-page-layout-1) for the markup.

- **Count:** keep to 3-6 so columns stay balanced. For an awkward count (e.g. 7), drop the least-actionable metric or merge two — don't leave an orphan on a partial row.
- **Priority order:** the most action-triggering metric goes leftmost; attention decays across the row.
- **The value is the hero** — largest element. Labels terse (2-4 words); longer explanation goes in a tooltip.
- **Pair each KPI with context** — "2,048 — 93% allocated" beats a bare "2,048".

### Chart grid

Charts sit below the stat blocks in a `Grid`, each in its own `Panel` (see the [example](#dashboard-page-layout-1)). Charts are sibling regions with different data — the Panel gives them visual boundaries.

- **No nested containment** — the `Panel` is the only container. Don't add an inner `div` with background/border/shadow.
- **Titles use manual `Text` (`body/bold/md`) consistently**
- **Hero chart spans columns** with `<GridItem cols={{ lg: 2 }}>`. Reserve it for a time-series trend that benefits from width.
- **Chart count:** 3-4 per page (the stat row doesn't count).

**Decluttering:**

- Remove gridlines if tooltips show exact values on hover
- Remove y-axis labels on sparklines and small charts
- Remove legends when the chart has only one series — the title is the legend
- Remove axis titles when the axis label is obvious from context

### Specialized panel patterns

#### Ranking list panel ("top consumers")

A list of label + value rows inside a Panel. Each row is a horizontal key-value pair with **explicit separation** between label and value.

```tsx
<Flex justify="between" align="baseline" gap="4">
  {/* justify="between" keeps the value column scannable — never render the
      row as one string ("Data Generation 288 GPUs") */}
  <Text kind="body/regular/md">{row.label}</Text>
  <Text kind="body/bold/md">{row.value}</Text>
</Flex>
```

#### Queue / follow-up panel (text + CTA)

A panel that summarizes a state and offers one action.

```tsx
<Panel>
  <Stack gap="1">
    <Flex justify="between" align="center" gap="4">
      <Text kind="body/bold/md">Follow-up queue</Text>
      <Button kind="tertiary">View queue</Button>
    </Flex>
    <Text kind="body/regular/md" className="text-secondary">
      4 items pending review.
    </Text>
  </Stack>
</Panel>
```

### Table section (optional)

When a dashboard includes a data table, add it as the last child of the page `Stack`, wrapped in a `Panel` titled with `Text kind="body/bold/md"`. Truncate long tables to ~10 rows with a "View all" link to the full list view.

---

## Responsive behavior

Responsiveness is declarative (see the [example](#dashboard-page-layout-1)):

- **Stat grid** reflows across breakpoints; keep at least 2 columns as soon as the viewport allows so the KPIs stay comparable.
- **Global filters on mobile:** collapse behind a "Filters" button if more than 2 filters; keep time range always visible.

---

## React examples

### Dashboard

Canonical dashboard body inside the stacked shell (`max-w-7xl` main, `padding-block: 1.5rem`) from layout-principles.md.

```tsx
<Stack gap="6">
  <PageHeader kind="flat" slotHeading="Cluster Utilization" slotDescription="GPU capacity, health, and allocation overview." />
  <Flex align="center" gap="6" className="w-full" wrap="wrap">
    <Flex gap="2" wrap="wrap">
      <Select attributes={{ SelectTrigger: { className: "w-44" } }} placeholder="Time range" defaultValue="24h" items={timeRangeItems} aria-label="Time range" />
      <Select placeholder="Team" attributes={{ SelectTrigger: { className: "w-44" } }} defaultValue="all" items={teamItems} aria-label="Team" />
    </Flex>

    <Flex gap="2" align="center">
      <Text kind="body/regular/sm" className="text-secondary">
        Last updated 12s ago
      </Text>
      <Button kind="tertiary">
        <Icon name="refresh" />
        Refresh
      </Button>
    </Flex>
  </Flex>
  <Banner kind="inline" status="warning">
    10 clusters report stale utilization metrics. Aggregates still include them; confirm freshness before acting.
  </Banner>

  {/* 170px is the label-wrap threshold — tune to your longest label. Prefer this over hard-coded grid-cols breakpoints. */}
  <Grid gap="6" colMinWidth="170px">
    {kpis.map((kpi) => (
      <Panel key={kpi.label}>
        <Stack gap="3">
          <Text kind="body/bold/md">{kpi.label}</Text>
          <Text kind="label/bold/3xl">
            {kpi.value}
            {kpi.denominator ? (
              <Text kind="label/regular/lg" className="text-placeholder">
                {kpi.denominator}
              </Text>
            ) : null}
          </Text>
          <Text kind="body/regular/sm" className="text-placeholder">
            {kpi.context}
          </Text>
        </Stack>
      </Panel>
    ))}
  </Grid>

  <Grid gap="6">
    <GridItem cols={{ lg: 2 }}>
      <Panel slotHeading={<Text kind="body/bold/md">Utilization</Text>}>
        <Stack gap="1">
          <Text kind="body/regular/md">Average GPU utilization across all clusters, 30-day trend.</Text>
          <div className="grid h-[307px] place-items-center rounded-md bg-surface-raised">Chart Placeholder - Replace Me</div>
        </Stack>
      </Panel>
    </GridItem>

    <Panel slotHeading={<Text kind="body/bold/md">Allocation by team</Text>}>
      <Stack gap="1">
        <Text kind="body/regular/md">Largest consumers in the current filter scope.</Text>
        <Stack gap="3">
          {ranking.map((row) => (
            <Flex key={row.id} justify="between" align="baseline" gap="4">
              <Text kind="body/regular/md">{row.label}</Text>
              <Text kind="body/bold/md">{row.value}</Text>
            </Flex>
          ))}
        </Stack>
      </Stack>
    </Panel>

    <Panel slotHeading={<Text kind="body/bold/md">Cluster health</Text>}>
      <Stack gap="1">
        <Text kind="body/regular/md">Distribution by operational status.</Text>
        <div className="grid h-[241px] place-items-center rounded-md bg-surface-raised">Chart Placeholder - Replace Me</div>
      </Stack>
    </Panel>
  </Grid>

  <Panel>
    <Stack gap="1">
      <Flex justify="between" align="center" gap="2">
        <Text kind="body/bold/md">Clusters needing attention</Text>
        <Button asChild kind="tertiary">
          <a href="/clusters-needing-attention">View all</a>
        </Button>
      </Flex>
      <Text kind="body/regular/md">Rows are truncated to keep the dashboard scannable; link to the full list view for the complete set.</Text>
      <Table className="overflow-x-auto" columns={["Cluster", "Issue", "Region", "Updated"]} rows={dashboardTableRows.slice(0, 10)} />
    </Stack>
  </Panel>
</Stack>
```
