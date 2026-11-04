# OtpInput

A segmented field for entering a one-time passcode or PIN. One real input backs every cell, so browser autofill, paste, and native validation all keep working.

## Notes

- Reach for OtpInput only for short fixed-length codes the user transcribes from elsewhere — SMS and authenticator verification, device pairing, PIN entry. For any other short string use TextInput.

- Wrap in FormField with `slotLabel` so the field is named, and state the expected length and format in helper text; the cells are decorative and are hidden from assistive technology.

- Set `type` to match the code the sender actually issues. Leave it `"numeric"` for SMS codes so mobile keyboards show a keypad.

- When using `autoSubmit`, warn about automatic submission in associated helper text before entry and keep a visible submit button.

- Do not add a paste button or split the code across several inputs; pasting, SMS autofill, and keyboard navigation already work on the single underlying input.

## Examples

### Basic Otp Input

Wrap in FormField so the field is named and the expected length is stated in helper text. The cells are decorative, so the label and help text are all a screen reader has to go on.

```tsx
<FormField name="code" slotHelp="Enter the 6-digit code we sent to your phone." slotLabel="Verification code">
  <OtpInput />
</FormField>
```

### Grouped Otp Input

`groupSize` chunks the cells to match how the code is written down. Grouping is presentational — the submitted value never contains the separator, and a code pasted with dashes is accepted.

```tsx
<FormField name="code" slotLabel="Backup code">
  <OtpInput groupSize={3} length={6} separator="-" />
</FormField>
```

### Masked Otp Input

Masking suits a PIN the user re-enters in shared or screen-shared settings. Prefer an unmasked field for a code the user is transcribing once — they need to check it against the message.

```tsx
<FormField name="pin" slotLabel="PIN">
  <OtpInput defaultValue="4821" length={4} mask />
</FormField>
```

### Alphanumeric Otp Input

Set `type` to match the code your service actually issues; it drives the mobile keyboard, the `pattern` the browser validates against, and which characters are accepted.

```tsx
<FormField name="code" slotHelp="Letters and numbers, case-insensitive." slotLabel="Recovery code">
  <OtpInput length={8} type="alphanumeric" uppercase />
</FormField>
```

### Auto Submit Otp Input

Warn about automatic submission in associated helper text before entry, and keep a visible submit button for manual submission.

```tsx
() => {
  const [value, setValue] = useState("");
  const [verified, setVerified] = useState("");

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        setVerified(value);
      }}
    >
      <Stack gap="density-md">
        <FormField name="code" slotHelp="Enter the 6-digit code. Verification starts automatically when all 6 digits are entered." slotLabel="Verification code">
          <OtpInput autoSubmit onValueChange={setValue} value={value} />
        </FormField>
        <Button type="submit">Verify</Button>
        {verified ? <Text>Verifying {verified}…</Text> : null}
      </Stack>
    </form>
  );
};
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| attributes | `{ Cell?: ComponentPropsWithoutRef<"div">; Cells?: ComponentPropsWithoutRef<"div">; Input?: ComponentPropsWithRef<"input">; Separator?: ComponentPropsWithoutRef<"span"> }` | - | Native HTML attributes forwarded to the internal composed components. |
| autoSubmit | `boolean` | `false` | Calls `form.requestSubmit()` once the final character is entered. Warn about automatic submission in associated helper text and keep a visible submit button. |
| defaultValue | `string` | - | Uncontrolled initial value. Sanitized against `type` and `length` before it reaches the input. |
| groupSize | `number \| number[]` | - | Splits the cells into groups. A number gives equal chunks; an array gives explicit sizes. |
| length | `number` | `6` | Number of characters, and therefore cells. A value that is not a positive whole number is ignored and the default applies. |
| mask | `boolean` | `false` | Obscures the value, painting a bullet in each filled cell. |
| onComplete | `(value: string) => void` | - | Called when the value reaches `length`, and again whenever a complete value is replaced by a different one. Never fires on mount or on blur. |
| onValueChange | `(value: string, event: React.ChangeEvent<HTMLInputElement, Element>) => void` | - | Called with the sanitized value whenever the user changes it. Pair with `value` for controlled state. A value the browser wrote before hydration — autofill, a bfcache restore — is adopted silently, so read `value` from the input or control it. |
| renderCell | `(cell: OtpCellState) => ReactNode` | - | Replaces the default cell rendering. Keep the `nv-otp-input-cell` class and `data-index={cell.index}` on the outer element so clicking a cell still places the caret; `attributes.Cell` applies to the default cell only. |
| separator | `ReactNode` | - | Rendered between groups. Omit for spacing alone. |
| size | `"small" \| "medium" \| "large"` | `"medium"` | Overall height and typography of the cells. |
| status | `"error" \| "success"` | - | The status of the input. Use `withValidation` to automatically apply success/error states based on `:user-valid` and `:user-invalid` pseudo classes. |
| type | `OtpInputType` | `"numeric"` | Accepted character set. Drives the `pattern` and `inputmode` attributes as well as filtering. |
| uppercase | `boolean` | `false` | Upper-cases typed and pasted characters. Has no effect when `type` is `"numeric"`. |
| value | `string` | - | Controlled value. Pair with `onChange` or `onValueChange` to keep it editable. |
| withValidation | `boolean` | - | Surfaces success and error styling automatically from the input's `:user-valid` and `:user-invalid` states. |

## Exports

- Components: `OtpCellState`, `OtpInput`, `OtpInputType`, `OtpSelection`
- Utilities: `charClassFor`, `filterOtpChars`, `inputModeFor`, `overwriteOtpValue`, `patternFor`, `resolveOtpCells`, `resolveOtpGroups`, `resolveOtpLength`, `sanitizeOtpValue`
- Constants: `DEFAULT_OTP_LENGTH`
- Types: `OtpInputProps`

## See also

- [FormField](./FormField.md)
- [TextInput](./TextInput.md)
