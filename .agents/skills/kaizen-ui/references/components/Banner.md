# Banner

A prominent, full-width inline message embedded in the page layout for system-wide announcements, page-level alerts, or important context.

## Notes

- Pick `kind` by message scope: `"global"` for app-wide announcements above the AppBar, `"header"` for page-level header alerts below the AppBar, `"inline"` for contextual alerts inside a content area.

- Banner spans full width and dominates the layout — use Notification (not Banner) for contextual floating messages related to an action.

- Always set `status` to match the message: `"error"` for outages or failures, `"warning"` for upcoming issues, `"success"` for resolved items, `"info"` for neutral announcements.

- Omit `onClose` for banners that must remain visible until resolved (active outages, compliance notices, security warnings). Provide `onClose` only when the user can safely dismiss the message.

- Choose the messaging component by attention and presence: Toast for low-attention transient confirmations, Notification for stacked persistent updates, Banner for page- or app-wide inline alerts, Modal for blocking high-attention decisions.

- Anti-patterns: using `kind="global"` for page-specific info; stacking multiple banners (causes alert fatigue — users stop reading); making active-outage banners dismissible.

- Dismissibility: informational announcements, feature promotions, and resolved status updates are safe to dismiss; active outages, compliance notices, and unresolved errors should omit `onClose`.

## Examples

### Inline Banner

Use `kind="inline"` (default) for contextual alerts within a specific content area. Lighter weight than `header` and the right choice for messages scoped to a single section.

```tsx
<Flex className="w-full" gap="1" direction="col">
  <Banner status="info">Lorem ipsum dolor sit</Banner>
  <Banner status="success">Lorem ipsum dolor sit</Banner>
  <Banner status="warning">Lorem ipsum dolor sit</Banner>
</Flex>
```

### Global Banner

Use `kind="global"` for app-wide announcements that render at the top of the app, above the AppBar — scheduled maintenance, account expiration, platform-level alerts that affect all pages.

```tsx
<Banner onClose={hideBanner} status="warning" kind="global">
  Your account password is expiring soon
</Banner>
```

### Header Banner With Actions

Use `kind="header"` for page- or section-level announcements that render below the AppBar. Header banners support heading + subheading + actions for context the current view needs.

```tsx
<Banner
  kind="header"
  onClose={hideBanner}
  slotSubheading="Our system will be unavailable on Saturday from 2-4 AM"
  slotActions={
    <>
      <Button kind="secondary" size="tiny">
        Support
      </Button>
      <Button kind="secondary" size="tiny">
        Add to Calendar
      </Button>
    </>
  }
>
  Service maintenance scheduled
</Banner>
```

### Composed

```tsx
<BannerRoot kind="global" status="warning">
  <BannerLayout role="group" aria-labelledby="maintenance-banner-heading">
    <BannerContent>
      <BannerIcon>
        <Warning />
      </BannerIcon>
      <BannerHeader>
        <BannerHeading id="maintenance-banner-heading">Service maintenance scheduled</BannerHeading>
      </BannerHeader>
    </BannerContent>
    <BannerActionsSection>
      <Button kind="secondary" size="tiny">
        Support
      </Button>
    </BannerActionsSection>
    <BannerCloseButtonSection>
      <Button aria-label="Dismiss banner" kind="tertiary" size="tiny">
        <Close />
      </Button>
    </BannerCloseButtonSection>
  </BannerLayout>
</BannerRoot>
```

### Composed Header Example

```tsx
<BannerRoot kind="header" status="success">
  <BannerLayout role="group" aria-labelledby="maintenance-complete-heading">
    <BannerContent>
      <BannerIcon>
        <SuccessIcon />
      </BannerIcon>
      <BannerHeader>
        <BannerHeading id="maintenance-complete-heading">Maintenance complete</BannerHeading>
        <BannerSubheading>Our system is back online</BannerSubheading>
      </BannerHeader>
    </BannerContent>
    <BannerActionsSection>
      <Button kind="secondary" size="tiny">
        Changelog
      </Button>
    </BannerActionsSection>
    <BannerCloseButtonSection>
      <Button aria-label="Dismiss banner" kind="tertiary" size="tiny">
        <Close />
      </Button>
    </BannerCloseButtonSection>
  </BannerLayout>
</BannerRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **kind** * | `"header" \| "inline" \| "global"` | `"inline"` | The kind of banner based on the scope of the message. - "inline" - Contextual alerts placed inline within a specific content area. - "header" - Page-level header banner with a bolded heading and optional subheading. Renders full width below the AppBar. - "global" - Site-wide announcements rendered full width at the top of the app, above the AppBar. |
| **slotSubheading** * | `ReactNode` | - | Subheading text rendered below the heading. Only applies to `kind="header"` banners. |
| actionsPosition | `"right" \| "bottom"` | - | Forces the actions to a specific position. By default, actions render on the right and move to the bottom when the banner is too narrow. - "right" - Actions stay to the right of the content. - "bottom" - Actions stack below the content. |
| attributes | `{ BannerContent?: NativeElementAttributes<"div", typeof BannerContent>; BannerHeader?: NativeElementAttributes<"div", typeof BannerHeader>; BannerHeading?: NativeElementAttributes<"div", typeof BannerHeading>; BannerSubheading?: NativeElementAttributes<"div", typeof BannerSubheading>; BannerIcon?: NativeElementAttributes<"div", typeof BannerIcon>; BannerCloseButton?: NativeElementAttributes<"button", typeof Button>; BannerLayout?: NativeElementAttributes<"div", typeof BannerLayout>; BannerActionsSection?: NativeElementAttributes<"div", typeof BannerActionsSection>; BannerCloseButtonSection?: NativeElementAttributes<"div", typeof BannerCloseButtonSection> }` | - | Native HTML attributes forwarded to the internal composed components. |
| children | `ReactNode` | - | Heading text rendered at the top of the banner. |
| onClose | `() => void` | - | Dismiss handler. When provided, a close button is rendered; omit for blocking banners (e.g. active outages, compliance notices) that must remain visible until resolved. |
| ref | `Ref<HTMLDivElement>` | - |  |
| slotActions | `ReactNode` | - | Action buttons rendered next to or below the banner content. |
| slotIcon | `ReactNode` | `status === "info" ? ( <Icon name="info-circle" variant="fill" /> ) : status === "warning" ? ( <Icon name="warning" variant="fill" /> ) : status === "error" ? ( <Icon name="error" variant="fill" /> ) : status === "success" ? ( <Icon name="check-circle" variant="fill" /> ) : null` | Icon rendered alongside the heading. Defaults to a status-appropriate icon; pass `null` to render no icon. |
| status | `"error" \| "warning" \| "success" \| "info"` | `"info"` | The semantic status of the banner. - "info" - Neutral information (blue). - "warning" - Potentially unwanted outcomes or upcoming issues (yellow). - "error" - A problem that needs attention, such as outages or failures (red). - "success" - A successful action or resolved issue (green). |

`* = required prop`

## Exports

- Components: `Banner`, `BannerActionsSection`, `BannerCloseButtonSection`, `BannerContent`, `BannerHeader`, `BannerHeading`, `BannerIcon`, `BannerLayout`, `BannerRoot`, `BannerSubheading`
- Types: `BannerActionsSectionProps`, `BannerCloseButtonSectionProps`, `BannerContentProps`, `BannerHeaderProps`, `BannerHeadingProps`, `BannerIconProps`, `BannerLayoutProps`, `BannerProps`, `BannerRootProps`, `BannerSubheadingProps`

## See also

- [Notification](./Notification.md)
- [Toast](./Toast.md)
