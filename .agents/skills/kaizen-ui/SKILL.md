---
name: kaizen-ui
description: Advises on Kaizen UI (KUI) React components, design patterns, and page layouts for NVIDIA applications. Applies to all UI work in a KUI React application—building, modifying, or improving an interface, screen, page, or component—even when the user does not mention KUI, design systems, or best practices. Covers forms, filters, search, sort, cards, settings, navigation, detail and key-value displays, dashboards, charts and data visualization, stat blocks, ranked lists, activity feeds, loading, empty and error states, feedback, iconography, visibility, and page layout decisions.
metadata:
  library: "kaizen-ui"
  library_version: "1.19.0"
  type: "core"
  version: "5"
---

# Kaizen UI Foundations React docs

Official documentation for Kaizen UI Foundations (KUI), NVIDIA's design system.

KUI is built from theme-responsive design tokens, shared component CSS, and framework components. The React implementation composes KUI component primitives and shared CSS through typed React APIs. An optional Tailwind plugin exposes the same design tokens as utilities.

Read [CONFIGURATION.md](./references/guides/CONFIGURATION.md) when installing or configuring KUI React.

## Gotchas

- Use existing KUI components over custom implementations.
- Prefer semantic CSS tokens over primitives, for example `text-primary` over `text-gray-100`, so themes work automatically.
- Use sentence case except for acronyms. Do not use all-caps labels or emojis.

## Skill workflow

For UI work in a KUI application:

1. **Identify the page type.** Load the matching reference from [Page type routing](#page-type-routing) before placing content.
2. **Identify the patterns in play.** Load every applicable reference from [Pattern routing](#pattern-routing).
3. **Build outside-in.** Apply [page shell](./references/layouts/page-shell.md), then [layout principles](./references/layouts/layout-principles.md), then patterns and components.
4. **Apply the framework contract.** Use the documented React component and prop APIs. Read [COMPOSITION.md](./references/guides/COMPOSITION.md) when composing through `asChild` or component primitives.
5. **Load component references.** Read each selected component's generated reference under [Components](#components) before implementation. Each reference is generated from the component's source, so its props and exports are complete and current — treat them as authoritative and do not re-derive them from `node_modules` or package source. If something you need is genuinely absent from the reference, it is a documentation gap worth reporting.
6. **Verify the result.** Check semantic markup, accessible names, keyboard behavior, responsive layout, and loading, empty, error, and disabled states.
7. **Re-evaluate existing structure before extending it.** A new region, fourth column, repeated slot, or deeply nested wrapper can signal that the page needs restructuring. Read [page-modification.md](./references/layouts/page-modification.md).

## Page type routing

Determine the page type first; its layout guide drives downstream pattern and component decisions.

- [Browse: find, filter, compare, or select from a collection](./references/layouts/browse-page.md)
- [List view: filterable table or list of items](./references/layouts/list-view.md)
- [Detail: inspect, monitor, or manage one entity](./references/layouts/detail-page.md)
- [Dashboard: aggregated metrics, trends, progress, or health](./references/layouts/dashboard.md)
- [Form: create or edit an entity; input that gets submitted](./references/layouts/form-page.md)
- [Workspace: editor, AI, or command-center surface with a persistent task area](./references/layouts/workspace-page.md)
- [Settings or preferences](./references/patterns/settings.md) with [page shell](./references/layouts/page-shell.md)
- [Full-page error: terminal route failure such as 401, 403, 404, or 500](./references/layouts/error-page.md)

For a single entity, choose between a contextual panel, standalone page, or both with [detail-placement.md](./references/layouts/detail-placement.md).

## Pattern routing

Load every pattern guide whose feature appears on the page.

- [Navigation — application, section, breadcrumb, and content navigation](./references/patterns/navigation.md)
- [Forms — create, edit, configure, and submit](./references/patterns/forms.md)
- [Filters and sort controls](./references/patterns/filters.md)
- [Search input and query behavior](./references/patterns/search.md)
- [Tables, card grids, or lists of structured data](./references/patterns/data-display.md)
- [Key-value display — read-only labeled facts](./references/patterns/key-value-display.md)
- [Charts, graphs, gauges, and other data visualization](./references/patterns/data-visualization.md)
- [Stat blocks / KPI tiles](./references/patterns/stat-blocks.md)
- [Ranked lists — "top N by X" contributor panels](./references/patterns/ranked-lists.md)
- [Activity feed — chronological events, history, audit log, or timeline](./references/patterns/activity-feed.md)
- [Settings and persistent preferences](./references/patterns/settings.md)
- [Compare flows — select one alternative](./references/patterns/data-display.md#single-selection-among-alternatives-compare-flows)
- [Entity cards — content and interaction inside cards](./references/patterns/entity-cards.md)
- [Toasts, notifications, banners, and durable feedback](./references/patterns/feedback.md)
- [Loading and progress states](./references/patterns/loading.md)
- [Empty states](./references/patterns/empty-states.md)
- [Error states](./references/patterns/error-states.md)
- [Confirmations for destructive or irreversible actions](./references/patterns/confirmations.md)
- [Element visibility — hide, read-only, disable, or enable](./references/patterns/element-visibility.md)
- [Icons — when to use them and which icon fits an action](./references/ICONS.md)

## Custom work with KUI

Reuse KUI components and tokens before introducing a parallel visual language.

- Prefer extending existing components or composing their documented subcomponents before creating a new primitive.
- Read [COMPOSITION.md](./references/guides/COMPOSITION.md) before using `asChild` or replacing a component's rendered element.
- Token references name tokens in their Tailwind spelling. Use those utilities when the application configures `@nvidia/foundations-tailwind-plugin`; otherwise write CSS against the matching variable. Check the application's stylesheet for the plugin import to see which applies.
- Prefixes map one to one: `bg-*` → `--background-color-*`, `text-*` → `--text-color-*`, `border-*` → `--border-color-*`, `rounded-*` → `--radius-*`, `shadow-*` → `--shadow-*`, `gap-*` and `p-*` → `--spacing-*`.
- Reserve inline `style` for runtime values and documented custom properties. Never inline a static value that a utility or a stylesheet rule can carry.
- Load the relevant token references:
  - [Colors, backgrounds, borders, and accents](./references/design-tokens/COLORS.md)
  - [Chart and data-visualization colors](./references/design-tokens/CHART_DATA_COLORS.md)
  - [Raw color primitives — last resort and not theme-aware](./references/design-tokens/COLOR_PRIMITIVES.md)
  - [Typography](./references/design-tokens/TYPOGRAPHY.md)
  - [Spacing, padding, and gap](./references/design-tokens/SPACING_SCALE.md)
  - [Shadows, breakpoints, container widths, transitions, and animations](./references/design-tokens/EFFECTS_AND_LAYOUT.md)

## Components

Load every component reference you use for exact imports, props, composition, examples, and related components.

### Actions

- [Anchor](references/components/Anchor.md) - Interactive text that navigates the user to another page, section, or resource.
  - sometimes known as Link
- [Button](references/components/Button.md) - A clickable element that triggers an action. Use specific verb + noun labels instead of vague text.
- [ButtonGroup](references/components/ButtonGroup.md) - A row of related buttons presented as a single set of actions with consistent styling.
- [Command](references/components/Command.md) - A modal menu that lets users search for and run actions or jump to pages from the keyboard. Opens over the page, filters its items as the user types, and closes when an item is chosen.
- [Dropdown](references/components/Dropdown.md) - A button that opens a menu of actions, links, or selectable options. Use for contextual actions, filters, and overflow menus.
- [Menu](references/components/Menu.md) - A list of actions, toggles, or selections. Use when the list should always be visible (sidebars, settings panels).
- [SegmentedControl](references/components/SegmentedControl.md) - A horizontal group of mutually exclusive options where selection takes effect immediately. Use for switching between alternate views or modes of the same content.
- [Switch](references/components/Switch.md) - A two-state toggle for binary settings whose change takes effect immediately. Commonly used for preferences and feature toggles.
  - sometimes known as Toggle, ToggleSwitch
- [ToggleButton](references/components/ToggleButton.md) - A button that switches an option on or off and keeps showing which state it is in. Use for a state that persists after the click, such as pinning, starring, or bold text.
  - sometimes known as Toggle, Pressed Button, Stateful Button
- [ToggleButtonGroup](references/components/ToggleButtonGroup.md) - A set of related toggle buttons that each switch an option on or off, with one tab stop for the whole set. Use for options that apply to the same content, such as text formatting or which panels are visible.

### Chat

- [ChatBubble](references/components/ChatBubble.md) - The surface of one chat message: a padded, rounded block for a person's words, or flat text for an AI reply.

### Data

- [Avatar](references/components/Avatar.md) - A small visual identifier for a user, account, or entity. Prioritizes an image when available and falls back to initials or an icon otherwise.
- [Badge](references/components/Badge.md) - A small color-coded label that displays status, category, or metadata at a glance. Badges are non-interactive and convey information rather than trigger actions.
- [CodeSnippet](references/components/CodeSnippet.md) - Displays source code with syntax highlighting and a copy button. Pick `kind="inline"` for code samples within prose and `kind="block"` for standalone, multi-line samples.
- [Kbd](references/components/Kbd.md) - A visual representation of a keyboard key or shortcut. Place it beside the control or instruction it describes.
- [KeyValueGroup](references/components/KeyValueGroup.md) - A description list of labeled values, such as the properties of one resource. Renders a native `<dl>`.
- [Legend](references/components/Legend.md) - A key mapping colors to the data series they represent.
- [List](references/components/List.md) - Groups related items into a sequence, supporting numbered, bulleted, and icon-marked formats.
- [Meter](references/components/Meter.md) - A bar showing how much of a bounded resource is in use.
- [SegmentedMeter](references/components/SegmentedMeter.md) - A bar showing how a bounded quantity divides across categories.
- [Severity](references/components/Severity.md) - A ranked severity level shown as a fixed icon and label pairing. Use it to communicate urgency, risk, or priority on issues, alerts, and incidents.
- [StatusIndicator](references/components/StatusIndicator.md) - A small colored dot that signals a new change or short-lived status.
- [Table](references/components/Table.md) - Organizes sets of structured data into columns and rows so users can scan, compare, and analyze information.
- [Tag](references/components/Tag.md) - A small label that categorizes or groups an item with text and optional icons. Use for classification and filtering; reach for Badge to communicate status.
- [Text](references/components/Text.md) - Renders text with the design system's typography tokens applied.

### Disclosure

- [Accordion](references/components/Accordion.md) - A vertically stacked group of sections the user can expand and collapse to show or hide their contents.
- [AnimatedChevron](references/components/AnimatedChevron.md) - A chevron icon that animates between two orientations to indicate the open or closed state of a disclosure. Use inside expand and collapse triggers when users need a visual cue for the current state.
- [Collapsible](references/components/Collapsible.md) - An unstyled disclosure that toggles a single block of content between expanded and collapsed states. Use to hide secondary detail behind a clickable summary so users can reveal more on demand.

### Feedback

- [Banner](references/components/Banner.md) - A prominent, full-width inline message embedded in the page layout for system-wide announcements, page-level alerts, or important context.
- [Notification](references/components/Notification.md) - A non-blocking overlay message that delivers detailed feedback in the user's peripheral vision, optionally with actions. Reach for it when the message needs more detail than a Toast or an action the user should take.
- [StatusMessage](references/components/StatusMessage.md) - A full-area centered message displayed when there is no content to show — empty states, error pages, permission denied, not found. Replaces the content area rather than overlaying it.
- [Toast](references/components/Toast.md) - A small, temporary overlay that gives the user non-critical feedback after an action. Reach for it when the message is informational and can be missed without consequence.
- [Toaster](references/components/Toaster.md) - Renders and manages the app's toast and notification stacks. Render one near the app root, then call `toast()` or `notification()` from anywhere.

### Forms

- [Checkbox](references/components/Checkbox.md) - A form control that lets users toggle a single option on or off. Supports a mixed state for representing partially-selected groups of related options.
- [Combobox](references/components/Combobox.md) - A text input paired with a filterable listbox for selecting one or more values from a list. Use when users need to search through a large set of options.
- [DatePicker](references/components/DatePicker.md) - An input field paired with a calendar popup for selecting a single date or a date range. Use when users need to enter or browse to a specific date.
- [FormField](references/components/FormField.md) - Groups a label, input, and helper text to keep form elements consistent and accessible.
  - sometimes known as Form Group, Fieldset
- [InputShell](references/components/InputShell.md) - A low-level wrapper that styles any element as a KUI input. Reach for the higher-level inputs first; use this only when composing a custom control.
- [Label](references/components/Label.md) - A text label for a form control. Prefer FormField for end-user forms; this primitive is intended for composing custom field layouts.
- [NumberInput](references/components/NumberInput.md) - A TextInput preset to `type="number"` for numeric-only entry.
- [OtpInput](references/components/OtpInput.md) - A segmented field for entering a one-time passcode or PIN. One real input backs every cell, so browser autofill, paste, and native validation all keep working.
- [RadioGroup](references/components/RadioGroup.md) - A set of mutually exclusive options where the user picks exactly one. Use when the available options can comfortably fit on screen at once.
- [RangeSlider](references/components/RangeSlider.md) - A two-thumb slider for selecting a minimum and maximum value within a range. Use for filtering by price, time window, or any min/max selection.
  - sometimes known as Range Input, Input Range
- [Rating](references/components/Rating.md) - A row of icons that shows a value within a range, or lets the user pick one.
- [Select](references/components/Select.md) - An input that lets the user pick one or more values from a predefined list.
- [Slider](references/components/Slider.md) - A single-value range input rendered along a bar. Use for continuous settings like volume, brightness, or threshold values where the relative position carries meaning.
  - sometimes known as Range Input, Input Range
- [TextArea](references/components/TextArea.md) - A multi-line text input for free-form responses such as descriptions, comments, or notes.
- [TextInput](references/components/TextInput.md) - A single-line input for text, numbers, or special symbols. Commonly used in forms; switch to TextArea when the response is likely to wrap.
- [TimeInput](references/components/TimeInput.md) - A TextInput preset to `type="time"` for time-of-day entry, including the built-in native time-picker button.
- [Upload](references/components/Upload.md) - A file picker with drag-and-drop support that renders previews and basic management controls for the selected files.

### Layout

- [Block](references/components/Block.md) - A general-purpose block-level container for arbitrary content. Use it as a low-level layout primitive when consistent spacing and overflow behavior are needed without writing custom CSS.
- [Divider](references/components/Divider.md) - A horizontal or vertical line that visually separates groups of content. Use to create breaks between sections, list items, or toolbar regions where a heading or whitespace alone would be too subtle.
- [Flex](references/components/Flex.md) - A layout primitive that arranges its children with CSS flexbox using design-token spacing.
- [Grid](references/components/Grid.md) - A two-dimensional layout primitive that arranges children into rows and columns. Use for dashboards, card listings, and any layout that needs both horizontal and vertical alignment.
- [Group](references/components/Group.md) - A container that joins related items into a single visual unit. Reach for it when adjacent elements should read as one cohesive cluster, such as chips, badges, or button rows.
- [Inline](references/components/Inline.md) - An inline layout primitive that renders a `<span>` with the system's spacing tokens exposed as props.
- [Resizable](references/components/Resizable.md) - A layout of panels separated by draggable edges, such as a sidebar beside main content or an editor above a terminal.
- [Stack](references/components/Stack.md) - A flex container that arranges its children in a column by default, with an optional divider between items.

### Loading

- [ProgressBar](references/components/ProgressBar.md) - Visual indicator for long-running operations like uploads, downloads, and other loading states.
- [Skeleton](references/components/Skeleton.md) - A placeholder block that mimics the shape of content while it loads. Match each skeleton's size and shape to the real content as closely as possible to keep layout shift minimal.
- [Spinner](references/components/Spinner.md) - An indicator for short waits with unknown progress.

### Navigation

- [AppBar](references/components/AppBar.md) - The top-level masthead for an application, anchoring branding, primary navigation, and global actions. Every application has exactly one.
  - sometimes known as Masthead, Navbar, Navigation Bar
- [Breadcrumbs](references/components/Breadcrumbs.md) - A navigation trail that shows the user's location within the app hierarchy and links back to parent pages. Use concise labels for each level.
- [HorizontalNav](references/components/HorizontalNav.md) - A horizontal navigation bar that organizes links to top-level sections of an application or site.
- [Pagination](references/components/Pagination.md) - Navigation control that splits a long list of items across multiple pages. Pair with tables and grids that exceed a single page of content.
- [Stepper](references/components/Stepper.md) - A progress tracker for moving through a fixed, ordered set of steps.
- [Tabs](references/components/Tabs.md) - Organizes related content into sections that share the same context and switches between them in place. Use for managing settings, comparing data, or any group of mutually exclusive views the user moves between without leaving the page.
- [TreeNav](references/components/TreeNav.md) - A hierarchical navigation list that lets users move through links at different levels, similar to a directory tree
- [VerticalNav](references/components/VerticalNav.md) - A vertical navigation menu of single links and nested link groups, typically rendered in a sidebar. Use as section-level navigation paired with a HorizontalNav for app-wide chrome, or as the sole navigation for apps with deep hierarchies and few global destinations.

### Overlays

- [Modal](references/components/Modal.md) - A `<dialog>` that interrupts the page to capture the user's attention until they act on or dismiss its content. Use for confirmations, focused tasks, and information the user must acknowledge before continuing.
- [Popover](references/components/Popover.md) - A floating panel that displays rich content, options, or actions anchored to a trigger.
- [SidePanel](references/components/SidePanel.md) - A `<dialog>` panel that slides in from the left or right edge to show supplementary content without leaving the current view.
- [Tooltip](references/components/Tooltip.md) - Reveals a short, supplemental label when the user hovers or focuses an element. Use to clarify icon-only controls or surface non-essential context.

### Surfaces

- [Card](references/components/Card.md) - A container that groups content, actions, and optional media about a single subject. Cards are fluid and grow to fill their container.
- [Carousel](references/components/Carousel.md) - A horizontal container that cycles through a series of content items.
- [Halo](references/components/Halo.md) - A decorative rotating gradient ring that wraps a single element to draw attention to it. Halo is presentational only — it adds no semantics, so the element inside keeps its own role, label, and behavior.
- [Hero](references/components/Hero.md) - A large page banner that combines headline messaging and calls to action with imagery.
- [Panel](references/components/Panel.md) - Container that groups related content and actions with a background, border, and consistent padding. Use for general-purpose surfaces on dashboards and detail pages.
- [PageHeader](references/components/PageHeader.md) - Top-of-page banner that establishes context with a title, breadcrumbs, and page-level actions.

### Utilities

- [PortalProvider](references/components/PortalProvider.md) - Sets default portal behavior for descendant components that support portaling. Wrap the app (or a region) to make overlays portal to a shared target without passing `portal`/`portalContainer` to each one; descendants still override with their own props.
- [ThemeProvider](references/components/ThemeProvider.md) - Establishes a theme context for descendant components, controlling color theme and density. Nest providers to scope different settings to specific parts of the app.
