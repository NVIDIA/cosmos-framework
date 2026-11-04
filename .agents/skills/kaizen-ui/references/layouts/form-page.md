# Form page layout

Use for pages where users provide information to create or edit an entity.

Also see [forms.md](../patterns/forms.md) for details on how forms should be structured.

## When to use

- User is creating a new item or editing an existing one
- Page collects input that will be saved
- There is a clear submit/save action
- User needs to complete the task before moving on

**Examples:** create task, edit deployment, configure model, onboarding flow, settings form.

---

## Page shell + structure

**Start with the page shell from [layout-principles.md](./layout-principles.md).** The `<main>` wrapper with responsive padding is required. Forms use a narrower max-width than browse/detail pages.

### Max-width selection

| Form                                                                        | Max-width                        | Notes                                                    |
| --------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------- |
| **Default**                                                                 | 56rem (896px)                    | Use for forms accessed from wide layouts (browse, list). |
| Simple forms (< 6 fields)                                                   | 48rem (768px)                    |                                                          |
| Operator-facing config (4+ sections, dense field groups, side-by-side rows) | 56rem to 64rem (896px to 1024px) |                                                          |
| Single-column upper bound                                                   | 64rem (1024px)                   | Beyond this, use two-column.                             |
| Two-column form                                                             | 64rem (1024px)                   |                                                          |

**Consistency rule:** all pages do not need the same max-width, but the transition should feel intentional. A form that narrows from an 80rem browse page to 48rem feels jarring. Default to 56rem for forms accessed from wide layouts.

### Input sizing

All form controls — `TextInput`, `Select`, `TextArea`, `Switch`, `Checkbox` — use the design system's **default size (medium)**. Do not mix sizes within a form.

### Entry point

Forms are **actions on entities**, not navigation destinations. A form page is reached from entity context — a row action in a browse table, a button in a detail page header, a "New [entity]" button in a list-view toolbar. **Do not add form pages as top-level navigation items.** See [navigation.md](../patterns/navigation.md).

### Back navigation — Breadcrumbs, not a Back button

Use [Breadcrumbs](../components/Breadcrumbs.md) above the page title for hierarchical back navigation; do not use a standalone Back button. The breadcrumbs and title belong in [PageHeader](../components/PageHeader.md).

A Cancel button in the form footer is **separate** from back-navigation: Cancel walks back from in-progress input (may discard unsaved data), Breadcrumbs walk the page hierarchy. Both can coexist. If the form opens in a modal/dialog, the modal's close affordance handles back-navigation — Breadcrumbs do not apply.

---

## Regions

### Header

- Breadcrumbs (no standalone "Back" button)
- Page title (action + entity: "Create Task", "Edit Deployment")
- Description (brief context about what the user is creating)

### Step indicator (multi-step only)

- Step labels (short: "Type", "Details", "Configuration", "Review")
- Current step indicator
- Completed step indicators

**Rules:**

- Keep step labels short (1–2 words)
- 3–5 steps max; more = reconsider structure
- Don't use a stepper for fewer than 3 steps

### Form content

Each `<section>` follows this composition:

- **Section title:** use the `title/sm` typography token.
- **Section description** (optional): use `body/md` with `text-secondary` in a block below the title.
- **Fields container:** use a vertical stack with a 12px within-section gap and 16px above the field group.
- **Trailing divider** between sections (omit on the last).

### Checkbox layout

See [forms.md](../patterns/forms.md) Decision 5.

### Actions (footer)

- Cancel (returns to list without saving)
- Submit / Next Step (primary action)
- If multi-step: Back on the left, Cancel + Submit on the right

---

## Containment

- **No panel** around the entire form in single-column layout
- **Panels** for branching-choice cards (bordered, selectable)
- **Disclosure sections** for optional or advanced configuration
- **Subtle background** for code input areas

---

## When NOT to use a form page

| Situation                      | Instead use                  |
| ------------------------------ | ---------------------------- |
| Single-field edit (rename)     | Inline edit or modal         |
| Quick action with confirmation | Modal dialog                 |
| Settings with immediate effect | Settings page with auto-save |
| Multi-select from a list       | Browse page with selection   |

---

## Review step (multi-step forms)

The final step before submit should:

- Show a summary of all entered values
- Group by section, matching form structure
- Allow edit via "Edit" links that return to the relevant step
- Make the submit action prominent

---

## React examples

### Sectioned form

Operator-facing form body with 3 sections. Shell uses `max-w-4xl` on `<main>` (layout-principles.md). Breadcrumb back navigation (no standalone Back button), block-displayed heading/description, `space-y-6` between sections, `Stack gap="3"` within sections.

```tsx
<form>
  <Stack gap="6">
    <PageHeader slotBreadcrumbs={<Breadcrumbs items={breadcrumbItems} />} slotHeading="Create cluster" slotDescription="Provision a new GPU cluster — choose region, hardware, and networking." />

    <Stack gap="3">
      <Text kind="body/bold/xl">Identity</Text>
      <FormField slotLabel="Cluster name" name="name">
        <TextInput placeholder="dfw-hgx-h100-prod" required />
      </FormField>
      <FormField slotLabel="Description" name="description">
        <TextArea placeholder="What will this cluster be used for?" />
      </FormField>
    </Stack>

    <Divider />

    <Stack gap="3">
      <Stack gap="1">
        <Text kind="body/bold/xl">Hardware</Text>
        <Text kind="body/regular/md" className="text-secondary">
          Pick the region, accelerator, and node count.
        </Text>
      </Stack>
      <FormField slotLabel="Region" name="region">
        <Select items={regionItems} />
      </FormField>
      <FormField slotLabel="GPU type" name="gpuType">
        <Select items={gpuItems} />
      </FormField>
      <Flex gap="inherit">
        <FormField slotLabel="Node count" name="nodeCount">
          <TextInput type="number" placeholder="32" />
        </FormField>
        <FormField slotLabel="GPUs per node" name="gpusPerNode">
          <TextInput type="number" placeholder="8" />
        </FormField>
      </Flex>
    </Stack>

    <Divider />

    <Flex gap="3" justify="end">
      <Button kind="secondary" type="button">
        Cancel
      </Button>
      <Button color="brand" type="submit">
        Create
      </Button>
    </Flex>
  </Stack>
</form>
```
