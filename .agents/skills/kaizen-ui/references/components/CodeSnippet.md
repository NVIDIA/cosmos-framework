# CodeSnippet

Displays source code with syntax highlighting and a copy button. Pick `kind="inline"` for code samples within prose and `kind="block"` for standalone, multi-line samples.

## Notes

- Always pass `language` so the highlighter picks the correct grammar — fall back to `"text"` only when the content is plain text.

- For long code blocks, enable `collapsible` and set `rows` to cap the visible height; otherwise the snippet expands inline and pushes surrounding content down.

- For server-rendered pages, pass `defaultValue` (the pre-highlighted HTML produced by `highlightCode`) so the snippet renders without waiting for Shiki to load on the client.

- SSR/utility helpers: use `initHighlighter()` (preload), `getHighlighter()` (access the singleton), and `highlightCode(code, language, theme, options?)` (produce HTML for `defaultValue`).

- For narrowing user-supplied language strings into `CodeSnippetLanguage`, use the type-guard helpers `isCodeSnippetLanguage(str)` and `getCodeSnippetLanguage(str, fallback)`.

- For custom action buttons, prefer `size="small"` so they match the built-in copy and expander buttons.

- Use `copyButton="inline"` for short snippets that would otherwise be dwarfed by the actions row; it falls back to the header when `collapsible` is enabled. Use `copyButton="none"` to omit the button entirely.

## Examples

### Basic Code Snippet

Use for displaying code blocks with syntax highlighting and a copy button.

```tsx
<CodeSnippet language="javascript" value="console.log('Hello, world!');" />
```

### Inline Code Snippet

Use when referencing code inline within a paragraph of text.

```tsx
<p>
  Use the <CodeSnippet kind="inline" value="useState" language="typescript" /> hook to manage state.
</p>
```

### Collapsible Code Snippet

Use when displaying long code that should be collapsed by default to save vertical space.

```tsx
<CodeSnippet value={longCode} language="typescript" collapsible rows={5} />
```

### With Inline Copy Code Snippet

Use for short snippets, such as a single install command, where a dedicated actions row would take more vertical space than the code. Resolves back to "header" when `collapsible` is set, because the expand toggle needs the actions row.

```tsx
<CodeSnippet value="pip install nvidia-nim" language="bash" copyButton="inline" />
```

### With Copy Callback Code Snippet

Use when you need to track or respond to copy events, such as for analytics.

```tsx
<CodeSnippet value={sampleCode} language="typescript" onCopySuccess={handleCopy} />
```

### With Custom Actions Code Snippet

Use when you need to add custom actions to the code snippet. For Buttons, use size="small" so they match the built-in copy and expander buttons.

```tsx
<CodeSnippet
  value={sampleCode}
  language="typescript"
  slotActions={
    <Button className="mr-auto" kind="secondary" size="small">
      Run
    </Button>
  }
/>
```

### With Fixed Theme

By default, the code snippet will render in the current theme. You can fix the theme to light or dark by applying the light or dark class to the CodeSnippet or its parent

```tsx
<div className="nv-dark bg-surface-base">
  <CodeSnippet value={sampleCode} language="typescript" />
</div>
```

### With Server Side Rendering Code Snippet

Use when you need to server-side render the code snippet.

```tsx
async () => {
  // use highlightCode util to server side highlight the code
  const prerenderedCode = await highlightCode(sampleCode, "typescript");
  return <CodeSnippet defaultValue={prerenderedCode} language="typescript" value={sampleCode} />;
};
```

### Composed

```tsx
<Flex>
  Block:
  <CodeSnippetRoot kind="block" collapsible rows={4}>
    <CodeSnippetActions>
      <Button kind="secondary" size="small">
        Custom Action
      </Button>
      <CodeSnippetCopyButton value={sampleCode} />
      <CodeSnippetExpanderButton />
    </CodeSnippetActions>
    <CodeSnippetCode value={sampleCode} language="typescript" />
  </CodeSnippetRoot>
  Inline:
  <CodeSnippetRoot kind="inline">
    <CodeSnippetCode value={sampleCode} language="typescript" />
  </CodeSnippetRoot>
</Flex>
```

## Props

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| **kind** * | `"inline" \| "block"` | `"block"` | Visual layout of the snippet. - "block" - Stacked block layout that hosts actions and a copy button. - "inline" - Inline span sized to fit within surrounding prose. |
| **value** * | `string` | - | Source code rendered inside the snippet. |
| attributes | `{ CodeSnippetActions?: NativeElementAttributes<"div", typeof CodeSnippetActions>; CodeSnippetCode?: NativeElementAttributes<"div", typeof CodeSnippetCode>; CodeSnippetCopyButton?: NativeElementAttributes<"button", typeof CodeSnippetCopyButton>; CodeSnippetExpanderButton?: NativeElementAttributes<"button", typeof CodeSnippetExpanderButton> }` | - | Native HTML attributes forwarded to the internal composed components. |
| collapsible | `boolean` | - | Adds a chevron toggle and hides overflow until the user expands the snippet. Only honored when `kind="block"`. |
| copyButton | `"header" \| "none" \| "inline"` | `"header"` | Whether the snippet renders a copy button, and where it sits. - "header" - In the actions row above the code, next to any custom actions. - "inline" - Pinned to the top-right corner of the code itself, so the snippet spends no vertical space on an actions row. - "none" - No copy button. The actions row still renders for `slotActions` or the expand toggle.  Falls back to "header" when `collapsible` is true, because the expand toggle needs the actions row anyway. Always resolves to "none" when `kind="inline"`, which has no code area to pin a button to. |
| defaultOpen | `boolean` | `false` | Initial open state when uncontrolled. |
| defaultValue | `string` | - | Pre-highlighted HTML used in place of client-side highlighting. Pass server-rendered HTML here for hydration-safe SSR. |
| language | `"html" \| "text" \| "go" \| "typescript" \| "javascript" \| "tsx" \| "jsx" \| "json" \| "css" \| "bash" \| "shell" \| "python" \| "rust" \| "yaml" \| "markdown"` | `"text"` | Language token used for syntax highlighting. |
| onCopySuccess | `() => void` | - | Called after the code is successfully copied to the clipboard. |
| onOpenChange | `(open: boolean) => void` | - | Called whenever the open state changes. |
| open | `boolean` | - | Controlled open state of the collapsible block. |
| rows | `number` | - | Number of code rows visible before scrolling or collapsing. Acts as the collapsed height when `collapsible` is true, otherwise caps the height with overflow scroll. Only honored when `kind="block"`. |
| slotActions | `ReactNode` | - | Additional action elements rendered alongside the built-in controls. |
| structure | `"inline" \| "classic"` | `Derived from `kind`: "block" -> "classic", "inline" -> "inline".` | HTML element structure used for rendering. Override when block-styled code must nest inside a `<p>` tag. - "classic" - Uses block elements (`<div>`, `<pre>`, `<code>`); cannot nest inside `<p>` tags. - "inline" - Uses inline elements (`<span>`); safe to nest inside `<p>` tags (e.g. Markdown). |

`* = required prop`

## Exports

- Components: `CodeSnippet`, `CodeSnippetActions`, `CodeSnippetCode`, `CodeSnippetCopyButton`, `CodeSnippetExpanderButton`, `CodeSnippetRoot`
- Utilities: `assertCodeSnippetLanguage`, `getCodeSnippetLanguage`, `getHighlighter`, `highlightCode`, `initHighlighter`, `isCodeSnippetLanguage`, `sanitizeCodeHtml`
- Constants: `DEFAULT_LANGUAGES`, `DEFAULT_THEMES`
- Types: `CodeSnippetActionsProps`, `CodeSnippetCodeProps`, `CodeSnippetCopyButtonProps`, `CodeSnippetExpanderButtonProps`, `CodeSnippetLanguage`, `CodeSnippetProps`, `CodeSnippetRootProps`, `CodeSnippetTheme`, `HighlightCodeOptions`
