# Stepper

A progress tracker for moving through a fixed, ordered set of steps.

## Notes

- Pass an `aria-label` that names the process, such as "Checkout progress". The stepper is exposed as a group and has no name of its own.

- When a step has `status="error"`, say what went wrong in its `slotDescription`. The default error icon is hidden from assistive technology.

## Examples

### Basic Stepper

Use to visually represent progress through a multi-step process like a checkout or wizard flow.

```tsx
<Stepper
  activeStep={1}
  aria-label="Checkout progress"
  items={[
    { slotHeading: "Cart", slotDescription: "Review your items" },
    { slotHeading: "Shipping", slotDescription: "Enter your address" },
    { slotHeading: "Payment", slotDescription: "Complete purchase" },
  ]}
/>
```

### Vertical Stepper

Use when steps have lengthy descriptions or content, or when the layout requires vertical stacking.

```tsx
<Stepper
  layout="vertical"
  activeStep={1}
  aria-label="Setup wizard"
  items={[
    { slotHeading: "Account", slotDescription: "Create your account" },
    { slotHeading: "Profile", slotDescription: "Set up your profile" },
    { slotHeading: "Review", slotDescription: "Confirm your details" },
  ]}
/>
```

### Compact Stepper

Use when space is limited and a minimal dot-based indicator is sufficient.

```tsx
<Stepper kind="compact" activeStep={1} aria-label="Upload progress" items={[{ slotHeading: "Upload" }, { slotHeading: "Process", slotDescription: "Step 2 of 3" }, { slotHeading: "Done" }]} />
```

### Completed Stepper

Use when all steps are complete by setting activeStep beyond the last index.

```tsx
<Stepper activeStep={3} aria-label="Completed flow" items={[{ slotHeading: "Step One" }, { slotHeading: "Step Two" }, { slotHeading: "Step Three" }]} />
```

### With Error Stepper

Use when a step has failed validation or encountered an error that needs user attention.

```tsx
<Stepper activeStep={1} aria-label="Deployment progress" items={[{ slotHeading: "Configure" }, { slotHeading: "Validate", status: "error" }, { slotHeading: "Deploy" }]} />
```

### With Custom Status Indicators Stepper

Use slotSuccessIndicator to override the completed-step indicator and slotErrorIndicator to override the error-step indicator.

```tsx
<Stepper
  activeStep={1}
  aria-label="Checkout with custom status indicators"
  items={[
    {
      slotHeading: "Account",
      slotDescription: "Signed in and verified.",
      slotSuccessIndicator: <i className="nv-icon nv-icon-check-circle" role="img" aria-label="Completed" />,
    },
    {
      slotHeading: "Shipping",
      slotDescription: "Address or delivery options need attention.",
      status: "error",
      slotErrorIndicator: <i className="nv-icon nv-icon-warning" role="img" aria-label="Needs attention" />,
    },
    {
      slotHeading: "Review",
      slotDescription: "Confirm and submit your order.",
    },
  ]}
/>
```

### With Content Stepper

Use slotContent to render arbitrary content (such as the form fields for the active step) directly under each step. Only renders in the default (non-compact) kind.

```tsx
<Stepper
  activeStep={1}
  aria-label="Onboarding"
  items={[
    {
      slotHeading: "Account",
      slotDescription: "Account created.",
    },
    {
      slotHeading: "Profile",
      slotDescription: "Tell us about yourself.",
      slotContent: <div>Profile form fields go here</div>,
    },
    {
      slotHeading: "Review",
      slotDescription: "Confirm your details.",
    },
  ]}
/>
```

### Clickable Nodes Stepper

Use slotStepIndicator to make step nodes interactive (links or buttons) so users can navigate between steps. Completed steps still show the check icon while keeping the indicator clickable underneath.

```tsx
<Stepper
  activeStep={1}
  aria-label="Navigable progress"
  items={[
    {
      slotHeading: "Cart",
      slotDescription: "Review your items.",
      slotStepIndicator: <a href="#cart">1</a>,
    },
    {
      slotHeading: "Shipping",
      slotDescription: "Enter your address.",
      slotStepIndicator: <a href="#shipping">2</a>,
    },
    {
      slotHeading: "Payment",
      slotDescription: "Complete your purchase.",
      slotStepIndicator: <a href="#payment">3</a>,
    },
  ]}
/>
```

### Compact Vertical Stepper

Use to combine the compact dot indicator with vertical stacking — fits narrow side panels or sidebars where horizontal space is limited.

```tsx
<Stepper
  layout="vertical"
  kind="compact"
  activeStep={1}
  aria-label="Upload progress"
  items={[{ slotHeading: "Upload", slotDescription: "Steps 1/3 completed." }, { slotHeading: "Process" }, { slotHeading: "Review" }]}
/>
```

### With Item Attributes Stepper

Use the per-item `attributes` prop to attach analytics tags, custom data attributes, or override accessibility props on a specific step.

```tsx
<Stepper
  activeStep={1}
  aria-label="Deployment progress"
  items={[
    {
      slotHeading: "Build",
      slotDescription: "Compile artifacts.",
      attributes: {
        StepperItem: {
          // @ts-expect-error - data attributes are not typed
          "data-analytics-id": "build-step",
        },
      },
    },
    {
      slotHeading: "Test",
      slotDescription: "Run integration tests.",
      attributes: {
        StepperItem: {
          // @ts-expect-error - data attributes are not typed
          "data-analytics-id": "test-step",
        },
      },
    },
    {
      slotHeading: "Deploy",
      slotDescription: "Release to production.",
      attributes: {
        StepperItem: {
          // @ts-expect-error - data attributes are not typed
          "data-analytics-id": "deploy-step",
        },
      },
    },
  ]}
/>
```

### Composed

```tsx
<StepperRoot activeStep={1} layout="horizontal" aria-label="Progress">
  <StepperItem value={0} slotHeading="Step One" />
  <StepperItem value={1} slotHeading="Step Two" slotDescription="Current step" />
  <StepperItem value={2} slotHeading="Step Three" />
</StepperRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **items** * | `{ slotHeading?: ReactNode; slotDescription?: ReactNode; slotContent?: ReactNode; slotStepIndicator?: ReactNode; slotSuccessIndicator?: ReactNode; slotErrorIndicator?: ReactNode; status?: "error" }[]` | - | Step definitions rendered in order. |
| activeStep | `number` | `0` | Zero-based index of the current step. Steps before it render as completed, the step at this index as active, and later steps as upcoming. |
| kind | `"default" \| "compact"` | `"default"` | Visual density of the stepper. `"compact"` collapses to a row of dots with the active step's text shown separately. |
| layout | `"horizontal" \| "vertical"` | `"horizontal"` | Direction the steps flow. |

`* = required prop`

## Exports

- Components: `Stepper`, `StepperItem`, `StepperRoot`
- Hooks: `useStepperContext`
- Types: `StepperItemProps`, `StepperProps`, `StepperRootProps`
