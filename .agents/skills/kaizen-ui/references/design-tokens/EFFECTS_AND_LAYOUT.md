# Effects and layout tokens

Shared primitive tokens for radii, border widths, breakpoints, container widths, shadows, transitions, and animations.

For spacing, padding, gap, and margin, see [SPACING_SCALE.md](../design-tokens/SPACING_SCALE.md) — it covers density tokens, the raw 4px scale, and the related Tailwind/component-prop surfaces.

## Radii

Static border-radius primitives. For radius that scales with the active density theme, use `rounded-density-xl` (see [SPACING_SCALE.md](../design-tokens/SPACING_SCALE.md)).

| Token            | Tailwind class              |
| ---------------- | --------------------------- |
| `--radius-none`  | `rounded-none`              |
| `--radius-sm`    | `rounded-sm`                |
| `--radius-md`    | `rounded-md`                |
| `--radius-lg`    | `rounded-lg`                |
| `--radius-xl`    | `rounded-xl`                |
| `--radius-2xl`   | `rounded-2xl`               |
| `--radius-3xl`   | `rounded-3xl`               |
| `--radius-round` | `rounded-round` (full pill) |

## Border widths

Tailwind: `border` (1px), `border-2`, `border-3`, `border-4`, `border-0`.

## Breakpoints

Min-width breakpoints for responsive utilities (`sm:`, `md:`, `lg:`, etc.). KUI uses the standard Tailwind set: `xs` 20rem, `sm` 36rem, `md` 48rem, `lg` 62rem, `xl` 75rem, `2xl` 100rem.

## Container widths

Max-width primitives — pair with `max-w-{size}` (e.g. `max-w-container-md`). Sizes step from `3xs` (16rem) through `7xl` (80rem) in roughly 4rem increments: `3xs` 16, `2xs` 18, `xs` 20, `sm` 24, `md` 28, `lg` 32, `xl` 36, `2xl` 42, `3xl` 48, `4xl` 56, `5xl` 64, `6xl` 72, `7xl` 80 (rem).

## Shadows

Three elevation steps. **In light mode, shadows are required to distinguish stacked surfaces** since adjacent surface tokens often share a color.

| Token         | Tailwind class | Use case                                         |
| ------------- | -------------- | ------------------------------------------------ |
| `--shadow-sm` | `shadow-sm`    | Subtle lift — raised cards on `bg-surface-base`. |
| `--shadow-md` | `shadow-md`    | Floating elements — popovers, dropdowns.         |
| `--shadow-lg` | `shadow-lg`    | Modals, dialogs, top-most overlays.              |

## Transitions and easing

Tailwind: `duration-150`, `duration-200`, `duration-250`, `duration-300`; `ease-out` for the standard curve.

## Animations

Tailwind: `animate-pulse`, `animate-spin`.
