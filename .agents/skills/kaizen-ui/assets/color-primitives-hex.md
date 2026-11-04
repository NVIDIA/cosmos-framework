# Color primitive hex values

Exact hex values for every NVIDIA brand color primitive and translucent overlay. This reference is shared data — for **when** to use primitives vs. semantic tokens, see [references/design-tokens/COLOR_PRIMITIVES.md](../references/design-tokens/COLOR_PRIMITIVES.md).

Reach for these values only when you need a specific hex (brand illustrations, glass/scrim effects on uncontrolled backgrounds). For ordinary UI work, use semantic tokens from [COLORS.md](../references/design-tokens/COLORS.md) instead — they theme automatically.

## Brand color scales

Each family ships an `050 → 950` scale (gray adds `000`, `025`, `950`, `975`, `1000`). Lower numbers are lighter; higher numbers are darker.

```css
--color-black: #000000;
--color-white: #ffffff;
--color-brand: #76b900;
--color-red-050: #ffe9e9;
--color-red-100: #ffd7d7;
--color-red-200: #ffbbbb;
--color-red-300: #ff8181;
--color-red-400: #fe3f3f;
--color-red-500: #e52020;
--color-red-600: #c21e1e;
--color-red-700: #961515;
--color-red-800: #650b0b;
--color-red-900: #4b0404;
--color-red-950: #2d0100;
--color-yellow-050: #feeeb2;
--color-yellow-100: #fcde7b;
--color-yellow-200: #f9c500;
--color-yellow-300: #ef9100;
--color-yellow-400: #df6500;
--color-yellow-500: #d73d00;
--color-yellow-600: #b93100;
--color-yellow-700: #8d2600;
--color-yellow-800: #601600;
--color-yellow-900: #441000;
--color-yellow-950: #2d0b00;
--color-green-025: #dafb7d;
--color-green-050: #cfff40;
--color-green-100: #bff230;
--color-green-200: #a5de15;
--color-green-300: #76b900;
--color-green-400: #549a00;
--color-green-500: #3f8500;
--color-green-600: #327100;
--color-green-700: #265600;
--color-green-800: #193800;
--color-green-900: #142700;
--color-green-950: #0d1a00;
--color-teal-050: #adfcf8;
--color-teal-100: #9aefe5;
--color-teal-200: #3ae3c9;
--color-teal-300: #1dbba4;
--color-teal-400: #139a86;
--color-teal-500: #0d8473;
--color-teal-600: #097061;
--color-teal-700: #04554b;
--color-teal-800: #033831;
--color-teal-900: #022723;
--color-teal-950: #011a19;
--color-blue-050: #cbf5ff;
--color-blue-100: #aaccee;
--color-blue-200: #7cd7fe;
--color-blue-300: #10b1fb;
--color-blue-400: #008af9;
--color-blue-500: #0074df;
--color-blue-600: #0060c7;
--color-blue-700: #0046a4;
--color-blue-800: #002781;
--color-blue-900: #002050;
--color-blue-950: #00112c;
--color-purple-050: #fae9ff;
--color-purple-100: #f9d4ff;
--color-purple-200: #f0b9fd;
--color-purple-300: #cd8ef0;
--color-purple-400: #c359ef;
--color-purple-500: #a846db;
--color-purple-600: #952fc6;
--color-purple-700: #741d9d;
--color-purple-800: #4d1368;
--color-purple-900: #331344;
--color-purple-950: #1f0f27;
--color-fuchsia-050: #ffe8f9;
--color-fuchsia-100: #ffd3f2;
--color-fuchsia-200: #feb5ee;
--color-fuchsia-300: #fc79ca;
--color-fuchsia-400: #e050b7;
--color-fuchsia-500: #d2308e;
--color-fuchsia-600: #b62475;
--color-fuchsia-700: #8c1c55;
--color-fuchsia-800: #5d1337;
--color-fuchsia-900: #420d25;
--color-fuchsia-950: #2d0919;
--color-gray-000: #ffffff;
--color-gray-025: #f7f7f7;
--color-gray-050: #eeeeee;
--color-gray-100: #e0e0e0;
--color-gray-1000: #000000;
--color-gray-200: #cccccc;
--color-gray-300: #a7a7a7;
--color-gray-400: #898989;
--color-gray-500: #757575;
--color-gray-600: #636363;
--color-gray-700: #4b4b4b;
--color-gray-800: #313131;
--color-gray-900: #222222;
--color-gray-950: #161616;
--color-gray-975: #0c0c0c;
```

## Translucent overlays

Black-on-X and white-on-X overlays for hover/pressed effects on imagery, glass-morphism, and scrim layers where transparency must survive uncontrolled backgrounds.

```css
--color-translucent-black-000: #00000000;
--color-translucent-black-050: #0000000d;
--color-translucent-black-100: #0000001a;
--color-translucent-black-120: #0000001f;
--color-translucent-black-150: #00000026;
--color-translucent-black-200: #00000033;
--color-translucent-black-300: #0000004d;
--color-translucent-black-400: #00000066;
--color-translucent-black-500: #00000080;
--color-translucent-black-600: #00000099;
--color-translucent-black-700: #000000b2;
--color-translucent-black-800: #000000cc;
--color-translucent-black-900: #000000e5;
--color-translucent-white-000: #ffffff00;
--color-translucent-white-050: #ffffff0d;
--color-translucent-white-100: #ffffff1a;
--color-translucent-white-120: #ffffff1f;
--color-translucent-white-200: #ffffff33;
--color-translucent-white-250: #ffffff40;
--color-translucent-white-300: #ffffff4d;
--color-translucent-white-400: #ffffff66;
--color-translucent-white-500: #ffffff80;
--color-translucent-white-600: #ffffff99;
--color-translucent-white-700: #ffffffb2;
--color-translucent-white-800: #ffffffcc;
--color-translucent-white-900: #ffffffe5;
```
