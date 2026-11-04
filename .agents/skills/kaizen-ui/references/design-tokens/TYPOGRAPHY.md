# Typography tokens

KUI implements the NVIDIA Brand typography system as named variants exposed through framework APIs and CSS utility classes. Pick a variant so font size, weight, and line height remain consistent. **Do not hand-tune `font-size`, `font-weight`, or `line-height` directly.**

## Picking a kind

Five categories. Pick by **purpose**, not by visual size.

| Category | Purpose                                | Sizes                            |
| -------- | -------------------------------------- | -------------------------------- |
| display  | Large hero headings and page titles.   | `2xl xl lg md sm xs`             |
| title    | Section headings and subheadings.      | `2xl xl lg md sm xs`             |
| body     | Paragraphs and main content.           | `3xl 2xl xl lg md sm xs`         |
| label    | UI labels, captions, dense small text. | `3xl 2xl xl lg md sm xs`         |
| mono     | Code snippets and monospace.           | `md sm` (and a few larger sizes) |

`body` and `label` also accept a **weight**: `regular`, `semibold`, `bold`, `light`, `italic`. A complete variant name looks like `body/bold/md` and maps to the `text-body-bold-md` CSS utility.


Pass the slash-separated name to the React `Text` component's `kind` prop.







`display` and `title` are bold-only — no weight in the name (`display/xl`, `text-title-lg`).

## Defaults when unsure

- Page heading → `display/xl` or `display/lg`.
- Section heading → `title/lg` or `title/md`.
- Paragraph copy → `body/regular/md`.
- Form labels and table headers → `label/regular/sm` or `label/semibold/sm`.
- Code → `mono/md`.

## Tailwind class shape

The utility class is the variant name with slashes replaced by dashes: `body/bold/md` becomes `text-body-bold-md`.

## Fonts and semantics


Use `Text asChild` around the heading element.


