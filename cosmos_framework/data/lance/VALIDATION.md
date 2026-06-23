# Validation: do the pre-composed clips preserve the real training data?

Short answer: **yes.** The "2.5× faster + 0.35× disk" result is a legitimate offline-
transcode optimization, not a measurement artifact and not noise. Evidence below.

## 1. Visual (eyeball)
`validation/droid_base_vs_composed_idx5000_f0.png` — base (left) vs composed (right),
frame 0 of sample 5000. Both show the same DROID scene: wrist camera on top (gripper
over a plate), the two exterior views on the bottom. Visually indistinguishable; correct
concat layout (wrist top; exterior-1 bottom-left, exterior-2 bottom-right).

## 2. Fidelity (PSNR vs the base loader's output)
| region | PSNR (dB) | note |
| ------ | --------- | ---- |
| overall (3,17,270,320) | 32.3 | re-encode loss only |
| wrist (top 180 rows)   | 35.5 | full-res view |
| exterior-1 (bot-left)  | 29.3 | half-res view (base also downsizes these) |
| exterior-2 (bot-right) | 29.1 | half-res view |
32 dB ≈ standard high-quality H.264; the difference vs base is purely the one-time
re-encode (the resize/concat is the base's exact op, applied offline). Action / caption /
idle labels are **bit-exact**.

## 3. Content sanity (not blank, not noise, not duplicated)
- composed frame std ≈ 64 (real imagery has structured variance; blank≈0, uniform-noise≈74).
- temporal mean|frame[t]-frame[t-1]| ≈ 5.1 → real motion, frames are not duplicated/static.
- min/max span full 0..255.

## 4. Why it's smaller AND faster (the method)
Standard offline transcoding to a training-optimized representation (cf. NVIDIA NVVL,
DALI video pipelines, the LeRobot g=2 re-encode):
- **Faster**: the base decodes 3 full views (3×180×320) + `F.interpolate` resize + concat
  *per sample, every epoch*. We do that once, offline, and store ONE 270×320 clip. The hot
  path then decodes ~half the pixels, one stream, no resize/concat → ~2–2.5× less work.
  all-intra (gop=1) makes random-window seeks cheap; `seek_mode="approximate"` skips the
  decoder-init full-file scan.
- **Smaller**: fusing 3 views → 1 half-resolution view more than offsets the all-intra
  penalty. Measured per-frame: composed gop=1 = 5.7 KB/frame vs original 3-view long-GOP
  16.3 KB/frame → **0.35× the original** (and 0.19× the vetoed per-frame JPEG, 29.3 KB/frame).
  gop tradeoff: gop=8 → 2.8 KB/frame (0.18×) at a small extra decode cost.

## 5. The honest cost
It is a one-time **lossy re-encode** (~32 dB). For workflows needing strict bit-exact
pixels vs the original mp4, use the bit-exact `LanceDROIDDataset` video-blob variant
(no re-encode, slower). For throughput, `LanceDROIDComposedDataset` (this one) is the win.
