# Color primitives (last resort)

Raw NVIDIA brand palette values shared by KUI implementations — fixed colors that **do not theme automatically**. Reach for these only when no semantic token in [COLORS.md](./COLORS.md) fits.

Common cases where primitives are appropriate:

- Brand expression that must stay a specific hue across light and dark modes.
- Background imagery and illustrations.

For everything else (UI surfaces, text, borders, feedback states, interactive elements), **use semantic tokens** — they adapt automatically and stay consistent with the rest of the system.

## Aliases

| Token           | Resolves to         |
| --------------- | ------------------- |
| `--color-white` | `--color-gray-000`  |
| `--color-black` | `--color-gray-1000` |
| `--color-brand` | NVIDIA green        |

## Usage forms

| Form           | Example                                           |
| -------------- | ------------------------------------------------- |
| Tailwind class | `bg-red-500`, `text-blue-300`, `border-green-700` |
| CSS variable   | `var(--color-red-500)`                            |

## Color scales

Available families: `red`, `yellow`, `green`, `teal`, `blue`, `purple`, `fuchsia`, `gray`. Each family ships an `050 → 950` scale (gray adds `000`, `025`, `950`, `975`, `1000`). Lower numbers are lighter; higher numbers are darker — pick by relative position (e.g. `bg-red-500` for a saturated mid-tone, `bg-red-100` for a tint).

For the exact hex of every primitive and the translucent overlay scales (`--color-translucent-black-*`, `--color-translucent-white-*`), load [`../../assets/color-primitives-hex.md`](../../assets/color-primitives-hex.md). Only do this when you need an exact hex — picking by name (`red-500` vs `red-700`) does not require the hex.
