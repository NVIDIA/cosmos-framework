# Upload

A file picker with drag-and-drop support that renders previews and basic management controls for the selected files.

## Notes

- Always specify `accept` to filter selectable file types, and surface the same constraint in children so users see it before opening the file dialog.

- Use `multiple` when more than one file is expected; omit it when the field should accept exactly one.

- Wrap in FormField via `renderInput` when the upload sits inside a form and needs a label, helper text, or validation chrome.

- Upload handles the UI layer only — file validation beyond MIME type filtering and the actual upload request are the application's responsibility.

## Examples

### Basic Upload

```tsx
<Upload />
```

### With Accept Upload

Use when only specific file types should be accepted, and display the constraint to users.

```tsx
<Upload accept="image/png,application/pdf">Accepts: image/png, application/pdf</Upload>
```

### Multiple Upload

Use when users need to upload more than one file at a time.

```tsx
<Upload multiple>Up to 50 MB per file</Upload>
```

### With Default Value Upload

Use to pre-populate the file list with already-uploaded or in-progress items, e.g. when editing a saved record or resuming an interrupted session.

```tsx
<Upload
  multiple
  defaultValue={[
    {
      id: "1",
      file: new File(["resume content"], "resume.pdf"),
      status: "success",
      uploadedBytes: 1024,
    },
    {
      id: "2",
      file: new File(["partial"], "cover-letter.pdf"),
      status: "uploading",
      uploadedBytes: 512,
    },
    {
      id: "3",
      file: new File(["broken"], "transcript.pdf"),
      status: "error",
      errorMessage: "Upload failed — please retry.",
    },
  ]}
/>
```

### Controlled Upload

Controlled mode lets you manage the selected files externally — useful for resetting after submission, syncing with form libraries, or showing custom UI based on file selection.

```tsx
() => {
  const [files, setFiles] = useState<FileUploadItem[]>([]);
  return <Upload multiple value={files} onValueChange={setFiles} />;
};
```

### With Custom Trigger Content Upload

Override `slotAnchor` and `slotHeaderText` to tailor the trigger copy, and use children to surface inline constraints alongside the call to action.

```tsx
<Upload multiple accept=".pdf,.doc,.docx" slotAnchor="Browse for documents" slotHeaderText=" or simply drag them in.">
  <span>Supported formats: PDF, Word</span>
  <span>·</span>
  <span>Maximum file size: 25 MB</span>
</Upload>
```

### With Callbacks Upload

Provide `onFileRetry` to expose a retry button on errored items, and `onFileRemove` to cancel in-flight requests or clean up server-side state when a file is removed.

```tsx
<Upload
  multiple
  defaultValue={[
    {
      id: "1",
      file: new File(["report content"], "report.pdf"),
      status: "error",
      errorMessage: "Network error — please retry.",
    },
  ]}
  onFileRetry={(item) => {
    console.log("Retry upload for", item.file.name);
  }}
  onFileRemove={(item) => {
    console.log("Cancel upload for", item.file.name);
  }}
/>
```

### Media List Upload

Use `listKind="media"` for image and video uploads where a compact thumbnail grid is preferred over the default card layout.

```tsx
<Upload
  multiple
  accept="image/*,video/*"
  listKind="media"
  slotAnchor="Add media"
  slotHeaderText=" or drag photos and videos here."
  defaultValue={[
    {
      id: "1",
      file: new File(["img"], "vacation.jpg", { type: "image/jpeg" }),
      status: "success",
    },
    {
      id: "2",
      file: new File(["vid"], "demo.mp4", { type: "video/mp4" }),
      status: "uploading",
      uploadedBytes: 1,
    },
  ]}
/>
```

### In Form Field Upload

Use `renderInput` to wrap the trigger in a `FormField` when the upload needs a label, help text, or needs to participate in a larger form layout.

```tsx
<Upload
  required
  renderInput={(trigger) => (
    <FormField name="documents" slotLabel="Upload documents" slotHelp="Files will be deleted after 30 days.">
      {trigger}
    </FormField>
  )}
/>
```

### Disabled Upload

Use when file upload should be temporarily unavailable based on form or permission state.

```tsx
<Upload disabled />
```

### With Validation Upload

Use when native validation cues should appear automatically via :user-valid and :user-invalid pseudo classes.

```tsx
<Upload required withValidation />
```

### Composed

Use composed primitives when you need full control over the trigger layout and file list rendering.

```tsx
<UploadRoot>
  <UploadTrigger>
    <UploadInputElement />
    <UploadDescription>Up to 50 MB</UploadDescription>
  </UploadTrigger>
  <UploadContent />
</UploadRoot>
```

### Button Trigger Upload

Use a single button instead of the drop zone where space is tight, such as a dialog or a comment box. `Button asChild` styles a `<label>` that wraps the input, so the button opens the file picker without JavaScript, takes keyboard focus through the input, and submits the files with the form.

```tsx
<UploadRoot multiple>
  <Button asChild kind="secondary">
    <label>
      <Icon name="paperclip" />
      Attach files
      <UploadInputElement accept=".pdf,.docx" name="attachments" />
    </label>
  </Button>
  <UploadContent />
</UploadRoot>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| accept | `string` | - | Comma-separated list of MIME types or file extensions the picker should allow. |
| attributes | `{ UploadDescription?: NativeElementAttributes<"div", typeof UploadDescription>; UploadInputElement?: NativeElementAttributes<"input", typeof UploadInputElement>; UploadContent?: NativeElementAttributes<"div", typeof UploadContent>; UploadItem?: NativeElementAttributes<"div", typeof UploadItem> }` | - | Native HTML attributes forwarded to the internal composed components. |
| children | `ReactNode` | - | Helper text rendered inside the trigger, such as accepted file types or size limits. |
| defaultValue | `{ id: string; file: File; errorMessage?: string; status: "error" \| "success" \| "uploading"; uploadedBytes?: number; hidePreview?: boolean } \| { id: string; file: File; errorMessage?: string; status: "error" \| "success" \| "uploading"; uploadedBytes?: number; hidePreview?: boolean }[]` | - | The value of the upload when initially rendered. Use for uncontrolled state. A form reset restores it. Its files are shown but not submitted with a form; only files the user picks or drops are. |
| disabled | `boolean` | `false` | Disables the upload, preventing file selection, drag-and-drop, and item actions. |
| kind | `"flat" \| "floating"` | `"flat"` | Visual treatment of the shell. `"flat"` has a border and background; `"floating"` is borderless and transparent. |
| listKind | `"media" \| "card"` | `"card"` | The layout used to display selected files. - "card" - Detailed list rows showing file name, size, and progress. Best for documents. - "media" - Compact thumbnail tiles. Best for images and videos. |
| multiple | `boolean` | - | Whether the upload allows multiple files. |
| onFileRemove | `(file: { id: string; file: File; errorMessage?: string; status: "error" \| "success" \| "uploading"; uploadedBytes?: number; hidePreview?: boolean }) => void` | - | Called when a file is removed. The file is removed from internal state regardless of whether this handler is provided. |
| onFileRetry | `(file: { id: string; file: File; errorMessage?: string; status: "error" \| "success" \| "uploading"; uploadedBytes?: number; hidePreview?: boolean }) => void` | - | Called when the retry button on a failed file is activated. Provide this handler to render the retry affordance. |
| onValueChange | `(file: { id: string; file: File; errorMessage?: string; status: "error" \| "success" \| "uploading"; uploadedBytes?: number; hidePreview?: boolean }) => void \| (files: { id: string; file: File; errorMessage?: string; status: "error" \| "success" \| "uploading"; uploadedBytes?: number; hidePreview?: boolean }[]) => void` | - | Callback triggered whenever the value of the upload changes. |
| renderInput | `(slotInput: ReactNode) => ReactNode` | - | Renders the trigger inside a custom wrapper. Useful for combining the input with a Tooltip or FormField. |
| showPreview | `boolean` | `true` | Whether to render a preview thumbnail for the file. |
| slotAnchor | `ReactNode` | - | Replaces the default "Choose file"/"Choose files" anchor label rendered inside the trigger. |
| slotHeaderText | `ReactNode` | `" or drag and drop it here."` | Trailing text shown next to the anchor that completes the call-to-action. |
| status | `"error" \| "success"` | - | The status of the input. Use `withValidation` to automatically apply success/error states based on `:user-valid` and `:user-invalid` pseudo classes. |
| value | `{ id: string; file: File; errorMessage?: string; status: "error" \| "success" \| "uploading"; uploadedBytes?: number; hidePreview?: boolean } \| { id: string; file: File; errorMessage?: string; status: "error" \| "success" \| "uploading"; uploadedBytes?: number; hidePreview?: boolean }[]` | - | The value of the upload. Must be used in conjunction with `onValueChange`. Only files the user picked or dropped are submitted with a form. |
| withValidation | `boolean` | - | Surfaces success and error styling automatically based on the inner input's `:user-valid` and `:user-invalid` states. |

## Exports

- Components: `Upload`, `UploadContent`, `UploadDescription`, `UploadInputElement`, `UploadItem`, `UploadItemActionsGroup`, `UploadRoot`, `UploadTrigger`
- Hooks: `useUploadContext`
- Contexts: `UploadContext`
- Types: `FileUploadItem`, `SingleOrMultipleUploadProps`, `UploadContentProps`, `UploadContextType`, `UploadDescriptionProps`, `UploadInputElementProps`, `UploadItemActionsProps`, `UploadItemProps`, `UploadProps`, `UploadRootProps`, `UploadTriggerProps`

## See also

- [FormField](./FormField.md)
