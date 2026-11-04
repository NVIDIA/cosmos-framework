# DatePicker

An input field paired with a calendar popup for selecting a single date or a date range. Use when users need to enter or browse to a specific date.

## Notes

- Always use DatePicker for date input — never a plain TextInput. Free-text date entry is error-prone, format-inconsistent, and lacks calendar context.

- Use `kind="range"` for start/end date pairs instead of two separate single DatePickers — the linked fields keep the range valid and visually connected.

- Constrain the valid window with `disabledMatcher` (e.g. `{ before: new Date() }` for future-only dates) and `startMonth` / `endMonth` for the navigable calendar.

- Sibling comparison: use DatePicker for any date selection, TextInput only for non-date text or with type="time" for time-of-day selection.

- For optional date fields, set `dismissible: true` so users can clear the selection. Leave it off on required fields — a required field must always carry a value, so the clear button creates a validation conflict.

- For `format`, follow the application's locale conventions; pair with `formatFn` for fully custom rendering.

## Examples

### Basic Date Picker

Use when users need to select a single date.

```tsx
<DatePicker />
```

### Range Date Picker

Set `kind="range"` when users need to select a start and end date, such as for filtering or scheduling.

```tsx
<DatePicker kind="range" />
```

### With Default Value Date Picker

Single pickers accept a `Date` for `defaultValue` / `value` and emit `Date | undefined` from `onValueChange`. Range pickers accept a `DateRange` object with optional `from` / `to` Date properties and emit `DateRange | undefined`. The value can become `undefined` when the user clears the field, so always treat the controlled value as nullable.

```tsx
<Flex direction="col" gap="2">
  <DatePicker defaultValue={today} />
  <DatePicker
    kind="range"
    defaultValue={{
      from: today,
      to: new Date(today.getTime() + 7 * 24 * 60 * 60 * 1000),
    }}
  />
</Flex>
```

### With Range Placeholder Date Picker

Pass an object `{ from, to }` to give the range fields distinct placeholders. A plain string is applied to both inputs.

```tsx
<DatePicker kind="range" placeholder={{ from: "Start date", to: "End date" }} />
```

### Disable Weekends Date Picker

Use `{ dayOfWeek: [...] }` to disable specific days, where 0 is Sunday and 6 is Saturday. The same matcher shape applies to range pickers.

```tsx
<DatePicker disabledMatcher={{ dayOfWeek: [0, 6] }} />
```

### Disable Past Dates Date Picker

Use `{ before }` to forbid dates earlier than a cutoff (here, today), or `{ after }` for the inverse.

```tsx
<DatePicker disabledMatcher={{ before: new Date() }} />
```

### Combined Disabled Matcher Date Picker

Pass an array to combine matchers; any date matched by _any_ entry is disabled. Other matcher shapes include a single `Date`, an array of `Date`s, a `{ from, to }` range, and a predicate `(date) => boolean`.

```tsx
<DatePicker kind="range" disabledMatcher={[{ dayOfWeek: [0, 6] }, { before: new Date() }]} />
```

### Constrained Range Date Picker

On a range picker `min` and `max` are the minimum and maximum number of **nights** in the selection — not absolute dates. HTML cannot couple two native date inputs, so these are only enforced by the enhanced widget; re-validate no-JS submissions with `validateDateRangeNights` on the server.

```tsx
<DatePicker kind="range" min={3} max={30} />
```

### Exclude Disabled Dates Range Date Picker

Range-only. With `excludeDisabledDates`, users cannot select a range that spans a disabled date. Without it, the range may straddle disabled dates as long as the endpoints themselves are selectable.

```tsx
<DatePicker kind="range" disabledMatcher={{ dayOfWeek: [0, 6] }} excludeDisabledDates />
```

### Calendar Bounds Date Picker

Use `startMonth` and `endMonth` to constrain the navigable calendar window (defaults: 100 years ago to 5 years from now). These values are also translated into native `min` / `max` for the no-JS fallback. Use `defaultMonth` to set the initially-visible month without selecting a date.

```tsx
<DatePicker startMonth={new Date(2024, 0, 1)} endMonth={new Date(2024, 11, 31)} defaultMonth={new Date(2024, 5, 1)} />
```

### Custom Format Date Picker

`format` accepts a `date-fns` token string (e.g. `"yyyy-MM-dd"`, `"dd-MM-yyyy"`, `"MMM d, yyyy"`). Display formatting only — the form value is always submitted as ISO `yyyy-MM-dd`.

```tsx
<DatePicker format="yyyy-MM-dd" />
```

### Custom Format Function Date Picker

Use `formatFn` for full control over display formatting; it overrides `format`. Always set `placeholder` when supplying `formatFn` because the format string can no longer be used as a default hint.

```tsx
<DatePicker
  formatFn={(date) =>
    date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    })
  }
  placeholder="Mon D, YYYY"
/>
```

### Time Zone Date Picker

Pass an IANA name (e.g. `"America/New_York"`) or UTC offset (e.g. `"-05:00"`) so the picker interprets and renders the date in that zone instead of the user's local zone.

```tsx
<DatePicker timeZone="America/New_York" defaultValue={new Date("2024-06-01T12:00:00Z")} />
```

### Form Submission Date Picker

Apply the form `name` to `DatePickerNativeFallback`, not to `DatePickerInput`. The enhanced text input is display-only; the hidden native `<input type="date">` is the element that participates in form submission and always submits an ISO `yyyy-MM-dd` value. Use `parseDateFromFormData(formData, "dueDate")` on the server.

```tsx
<DatePicker
  attributes={{
    DatePickerNativeFallback: { name: "dueDate" },
  }}
/>
```

### Form Submission Range Date Picker

Range pickers submit two independent ISO values, so set the `name` on both `DatePickerNativeFallbackFrom` and `DatePickerNativeFallbackTo`. Parse with `parseDateRangeFromFormData(formData, { from: "rolloutFrom", to: "rolloutTo" })`.

```tsx
<DatePicker
  kind="range"
  attributes={{
    DatePickerNativeFallbackFrom: { name: "rolloutFrom" },
    DatePickerNativeFallbackTo: { name: "rolloutTo" },
  }}
/>
```

### Custom Input Attributes Date Picker

Use the `attributes` API to set per-input `aria-label`, `autoComplete`, `data-*`, etc. without dropping to composed primitives. Single pickers expose `DatePickerInput`; range pickers expose `DatePickerInputFrom` and `DatePickerInputTo`.

```tsx
<DatePicker
  kind="range"
  attributes={{
    DatePickerInputFrom: {
      autoComplete: "off",
      "aria-label": "Trip start date",
    },
    DatePickerInputTo: {
      autoComplete: "off",
      "aria-label": "Trip end date",
    },
  }}
/>
```

### Composed

Use the composed primitives when you need full control over trigger, input, calendar layout, or to slot in a custom rendering between any of them.

```tsx
<DatePickerRoot kind="single">
  <DatePickerTrigger>
    <DatePickerInput />
    <DatePickerNativeFallback />
  </DatePickerTrigger>
  <DatePickerContent>
    <DatePickerCalendar />
  </DatePickerContent>
</DatePickerRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| align | `"center" \| "start" \| "end"` | `"center"` | Alignment of the popover relative to its anchor. |
| attributes | `{ DatePickerContent?: NativeElementAttributes<"div", typeof DatePickerContent>; DatePickerCalendar?: NativeElementAttributes<"div", typeof DatePickerCalendar>; DatePickerInput?: NativeElementAttributes<"input", typeof DatePickerInput>; DatePickerNativeFallback?: NativeElementAttributes<"input", typeof DatePickerNativeFallback> }` | - | Native HTML attributes forwarded to the internal composed components. |
| closeOnApply | `boolean` | `true` | Closes the DatePicker when the user applies a selection. |
| defaultMonth | `Date` | `The current month` | Initial visible month. |
| defaultOpen | `boolean` | `false` | Initial open state when uncontrolled. |
| defaultValue | `Date \| { from: Date; to?: Date; }` | - | Initial date when uncontrolled. |
| disabled | `boolean` | - | Disables interaction with the DatePicker. |
| disabledMatcher | `Date \| ((date: Date) => boolean) \| Date[] \| { from: Date; to?: Date; } \| { before: Date; } \| { after: Date; } \| { before: Date; after: Date; } \| { dayOfWeek: number \| number[]; } \| ((Date \| ((date: Date) => boolean) \| Date[] \| { from: Date; to?: Date; } \| { before: Date; } \| { after: Date; } \| { before: Date; after: Date; } \| { dayOfWeek: number \| number[]; } \| boolean)[]) \| boolean` | - | Matcher describing which dates are disabled. Accepts a single matcher or an array of matchers. |
| dismissible | `boolean` | - | Renders a clear button when the input has a value. |
| endMonth | `Date` | `Five years from today` | Latest month selectable in the calendar. |
| excludeDisabledDates | `boolean` | `false` | Prevents the user from selecting a range that spans any disabled dates. |
| format | `string` | `"M/d/yyyy"` | Date format string using `date-fns` syntax. Used for the displayed value and the default placeholder. Ignored when `formatFn` is provided. |
| formatFn | `(date: Date) => string` | - | Custom formatter for displayed dates. Takes precedence over `format`; pair with an explicit `placeholder` so the expected shape stays clear. |
| kind | `"single" \| "range"` | `"single"` | Selects single-date mode. |
| max | `number` | - | Maximum number of nights allowed in the range. |
| min | `number` | - | Minimum number of nights allowed in the range. |
| modal | `boolean` | `false` | Renders the popup as a modal that blocks interaction with the rest of the page. |
| month | `Date` | - | Controlled visible month. Pair with `onMonthChange`. |
| onCloseAutoFocus | `(event: Event) => void` | - | Called when focus returns to the trigger after the popover closes. Call `event.preventDefault()` to skip the default focus restoration. |
| onEscapeKeyDown | `(event: KeyboardEvent) => void` | - | Called when Escape is pressed inside the popover. Call `event.preventDefault()` to keep the popover open. |
| onMonthChange | `(month: Date) => void` | - | Called when the visible month changes through navigation or dropdown selection. |
| onOpenChange | `(open: boolean) => void` | - | Called when the open state changes. |
| onValueChange | `(value: Date) => void \| (value: { from: Date; to?: Date; }) => void` | - | Called when the user commits a new date. |
| open | `boolean` | - | Controlled open state. Pair with `onOpenChange`. |
| placeholder | `string \| string \| { from?: string; to?: string; }` | - | Placeholder shown when the input is empty. |
| portal | `boolean` | `false` | Renders the floating content in a React portal. |
| portalContainer | `HTMLElement` | `document.body` | Container for portaled floating content when no nearer portal target exists. |
| readOnly | `boolean` | - | Prevents the value from being changed while preserving form submission. |
| renderDropdown | `(props: RenderDropdownParams) => React.JSX.Element` | - | Custom renderer for the month and year selection dropdowns. |
| side | `"right" \| "bottom" \| "left" \| "top"` | `"bottom"` | Preferred side of the anchor to render against. Flips automatically when the popover would overflow the viewport. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Overall height and typography of the shell. |
| startMonth | `Date` | - | Earliest month selectable in the calendar. |
| status | `"error" \| "success"` | - | Visual validation state applied to the input. |
| timeZone | `string` | - | IANA time-zone name or UTC offset applied when parsing and formatting dates. |
| value | `Date \| { from: Date; to?: Date; }` | - | Controlled selected date. Pair with `onValueChange`. |

## Exports

- Components: `DatePicker`, `DatePickerCalendar`, `DatePickerCalendarDropdown`, `DatePickerContent`, `DatePickerInput`, `DatePickerNativeFallback`, `DatePickerRangeTrigger`, `DatePickerRangeTriggerField`, `DatePickerRoot`, `DatePickerTrigger`
- Utilities: `deriveNativeDateBounds`, `formatDate`, `formatDateForNativeInput`, `getCurrentDatePickerValue`, `getPlaceholderForField`, `getZonedTime`, `isDate`, `isDatePickerValueEqual`, `isDateRange`, `parseDate`, `parseDateFromFormData`, `parseDateRangeFromFormData`, `parseNativeDateInput`, `resolveNativeInputId`, `validateDateRangeConstraints`, `validateDateRangeNights`
- Constants: `DEFAULT_DATE_PICKER_FORMAT`, `NATIVE_DATE_INPUT_FORMAT`
- Types: `DatePickerCalendarDayProps`, `DatePickerCalendarDropdownProps`, `DatePickerCalendarProps`, `DatePickerContentProps`, `DatePickerInputProps`, `DatePickerNativeFallbackProps`, `DatePickerRangeTriggerFieldProps`, `DatePickerRangeTriggerProps`, `DatePickerRootProps`, `DatePickerTriggerProps`, `DatePickerType`, `DatePickerValue`, `DateRangeNightsValidationError`, `PlaceholderForType`, `RangeDatePickerProps`, `SingleDatePickerProps`, `ValueForType`

## See also

- [FormField](./FormField.md)
