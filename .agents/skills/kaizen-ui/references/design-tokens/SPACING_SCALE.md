# Spacing scale

The KUI design system's spacing scale powers every length-like value across implementations — margin, padding, gap, width, height, etc. Two value families share the scale:

1. **Density tokens** — preferred. Density-aware values scale with the active density theme, and follow a spacing scale chosen by UX guidance.
2. **Raw scale** — escape hatch. Fixed multiples of a 4px base unit.

## Density tokens

A semantic sizing scale that scales with the active density theme.

| Token         | Size |
| ------------- | ---- |
| `density-xxs` | 2px  |
| `density-xs`  | 4px  |
| `density-sm`  | 6px  |
| `density-md`  | 8px  |
| `density-lg`  | 12px |
| `density-xl`  | 16px |
| `density-2xl` | 24px |
| `density-3xl` | 32px |
| `density-4xl` | 48px |
| `density-5xl` | 64px |

When unsure, start with `density-md` for inner gaps and `density-lg` for section padding.

## Raw scale

Use the raw scale when geometry must remain fixed across densities, such as an icon or chart viewport. Prefer density tokens for layout spacing that should adapt with controls.

Each numeric token equals `n × 4px` (`"4"` → 16px). The scale also accepts `"0.25"`/`"0.5"`/`"0.75"` for 1-3px, the half-steps `"1.5"`/`"2.5"`/`"3.5"` for 6/10/14px, plus `"inherit"` and `"px"` (fixed 1px).

## Gotchas

- The literal string `"px"` is a fixed 1px value, **not** the same as `density-xxs`. Use `density-xxs` for hairline gaps that should follow the density theme.
- `"0"` collapses spacing entirely; for the smallest visible gap, use `density-xxs`.
