# Video input for the `reasoner` model-mode of inference — design

> **⚠️ SUPERSEDED (2026-06-08) — historical record.** Review found the inference
> processor is the repo `Qwen3VLProcessor` wrapper, which runs
> `do_sample_frames=False` and expects a **pre-decoded PIL frame list** (it drops
> forwarded kwargs) — not a path it samples itself. The shipped implementation
> therefore decodes frames via `torchvision.io.read_video` + Qwen `smart_nframes`
> and exposes **only `video_fps`** (the `num_frames`/`min_frames`/`max_frames`/
> `min_pixels`/`max_pixels` knobs described below were dropped). Treat the sections
> below as the original design intent; the code is the source of truth.

**Date:** 2026-06-07
**Branch:** `maoshengl/video_reasoner_inference`
**Status:** superseded by implementation (see banner above)

## Goal

Let `model_mode=reasoner` in the Cosmos inference engine
(`python -m cosmos_framework.scripts.inference`) accept a **local mp4 video**
as conditioning input, producing text that reasons over the clip — for both
`Cosmos3-Nano` and `Cosmos3-Super`. Today the reasoner accepts only a text
prompt or a single still image.

## Background: why this is a gap

The reasoner text-generation path runs entirely inside the Cosmos engine:

```
inference.py:_get_reasoner_sample_data        # loads ONE PIL image via Image.open
  -> OmniMoTModel.generate_reasoner_text      # builds {"type":"image",...} chat block
    -> net.generate_reasoner_text             # pass-through
      -> unified_mot._impl_generate_reasoner_text   # pixel_values + image_grid_thw only
        -> prepare_multimodal_reasoner_inputs       # image recipe only
```

Two hard blocks:

1. `_get_reasoner_sample_data` (`cosmos_framework/inference/inference.py`) calls
   `Image.open(vision_path)` unconditionally — PIL cannot decode mp4.
2. `_impl_generate_reasoner_text` and `prepare_multimodal_reasoner_inputs`
   **explicitly reject video** ("for I2V conditioning, frames must be passed as
   images" — they have no `pixel_values_videos` / `video_grid_thw` params).

Separately, `cosmos_framework/scripts/vlm/eval_videophy2.py` *does* consume
video, but through a **different, standalone path**: a raw HuggingFace
`Qwen3VLForConditionalGeneration` + `processor.apply_chat_template([{"type":
"video",...}])` + `model.generate()`. It never touches the Cosmos engine, so it
does not satisfy the goal of supporting `model_mode=reasoner` in
`scripts.inference`.

**Key enabling fact:** the vendored Qwen3-VL model under
`cosmos_framework/model/vfm/vlm/qwen3_vl/` already implements video end to end —
`get_video_features`, `get_rope_index(video_grid_thw=...)`,
`get_placeholder_mask(pixel_values_videos=...)`, a `video_token_id`, and a full
`video_processing_qwen3_vl.py`. Only the Cosmos reasoner **wrapper layers** are
hardcoded to images. So the change is additive plumbing, not new model logic.

## Approach (chosen)

**B1 — add a parallel video lane through the existing reasoner stack.**

Add optional video parameters alongside the existing image parameters through
the wrapper layers, leaving the image and text-only paths bit-identical. A given
prompt carries **either** an image, **or** a video, **or** neither — never both.
No mixed image+video support (not needed).

Approaches considered and rejected:

- **B2 — unify image+video into one "media item" abstraction.** Cleaner
  long-term and enables mixed media in one prompt, but larger blast radius, more
  validation/tests, and supports a capability not requested (YAGNI).
- **B3 — expose the HF `Qwen3VLForConditionalGeneration` route instead.** Bypasses
  the Cosmos engine entirely (no `model_mode=reasoner`, no parallelism /
  guardrails / output plumbing) — does not meet the goal.

## Data flow

```
inputs/reasoner/reasoner_video.json
  { model_mode: "reasoner", prompt, vision_path: "clip.mp4", video_*: ... }
        |
        v  args.py: vision_path resolves; extension -> ConditionVisionMode.VIDEO (already detected)
_get_reasoner_sample_data()
        |  detect .mp4 -> {prompt, "reasoner_videos": [path], "<video sampling kwargs>"}
        v                (vs "reasoner_images" for the image branch)
_generate_reasoner_batch()
        |  route videos -> model.generate_reasoner_text(videos=[...], video_* kwargs)
        v
OmniMoTModel.generate_reasoner_text(videos=..., video_* kwargs)
        |  build {"type":"video","video":path, <sampling kwargs>} chat block
        |  apply_chat_template -> pixel_values_videos, video_grid_thw
        v
net.generate_reasoner_text(pixel_values_videos=..., video_grid_thw=...)   [pass-through]
        v
unified_mot._impl_generate_reasoner_text(... video tensors ...)
        v
prepare_multimodal_reasoner_inputs(...)   NEW video branch:
        get_video_features -> get_placeholder_mask(video) -> get_rope_index(video_grid_thw)
        v
reasoner_forward -> AR decode -> text   (unchanged)
```

## Component changes

All new params are optional and default to `None`/absent, so existing callers
and the image/text-only paths are unchanged.

### 1. `qwen3_vl/utils.py` — `prepare_multimodal_reasoner_inputs` (the one real seam)

Add optional `pixel_values_videos` / `video_grid_thw` params. When they are set
(and the image params are not), run the video recipe using helpers that already
exist:

- `get_video_features(causal_lm, pixel_values_videos, video_grid_thw)` instead of
  `get_image_features`
- `get_placeholder_mask(..., video_features=video_embeds)` -> use the returned
  `_video_mask`
- `get_rope_index(..., video_grid_thw=video_grid_thw)` instead of the image grid

The `masked_scatter`, `visual_pos_masks`, deepstack alignment, and return shape
all stay identical — only which features and which grid feed in change. The
image branch is untouched. Update the docstring that currently says videos are
not supported.

### 2. `unified_mot.py` — `_impl_generate_reasoner_text`

Add `pixel_values_videos` / `video_grid_thw` params. Extend the pairing guard
(currently `(pixel_values is None) != (image_grid_thw is None)`) to also validate
the video pair and to reject image+video supplied together. Branch: if video
tensors present -> call `prepare_multimodal_reasoner_inputs` with them; else
existing behavior. Update the "Videos are not supported" docstring.

### 3. `unified_mot.py` + `cosmos3_vfm_network.py` — the two `generate_reasoner_text` pass-throughs

Add the two video params and forward verbatim. Pure plumbing.

### 4. `omni_mot_model.py` — `OmniMoTModel.generate_reasoner_text`

Add `videos: list[Any] | None = None` (parallel to `images`) plus the optional
video sampling kwargs (see schema below). Validate not-both (image and video).
When `videos` is set, build the last user message with a
`{"type": "video", "video": videos[idx], <sampling kwargs>}` block instead of the
image block, then read `pixel_values_videos` / `video_grid_thw` out of the
`apply_chat_template` output and pass them down. Same per-prompt `B=1` loop, same
CP/CFGP output broadcast.

### 5. `inference.py` — `_get_reasoner_sample_data` + `_generate_reasoner_batch`

- Builder: branch on `Path(vision_path).suffix`. Image extension keeps
  `Image.open` + `reasoner_images`. Video extension passes the **path string**
  under `reasoner_videos` (the processor decodes it — see "Frame sampling"
  below), and carries the resolved `video_*` sampling kwargs.
- Batch: read whichever key is present, apply the homogeneity check (no mixing
  within a batch), and call `generate_reasoner_text(videos=...)` with the
  sampling kwargs when videos are present.

### 6. `args.py` — schema (`SamplingArgs` / `SamplingOverrides`) + reasoner `sample_args.json`

Add the input-video sampling knobs. They are named with a `video_` prefix to
avoid colliding with the existing **output**-oriented `fps` / `num_frames`
fields (which mean output rate/length and are otherwise unused by the reasoner).

| New reasoner sample-arg | Maps to processor kwarg | Default     |
| ----------------------- | ----------------------- | ----------- |
| `video_fps`             | `fps`                   | `None` (->2)|
| `video_num_frames`      | `num_frames`            | `None`      |
| `video_min_frames`      | `min_frames`            | `None` (->4)|
| `video_max_frames`      | `max_frames`            | `None`(->768)|
| `video_min_pixels`      | `min_pixels`            | `None`      |
| `video_max_pixels`      | `max_pixels`            | `None`      |

`None` means "use the processor default," so the no-override behavior is
identical to relying purely on processor defaults. Only non-`None` values are
forwarded into the video block / processor kwargs.

## Frame sampling

The Qwen3-VL processor decodes the mp4 and samples frames itself; we pass the
**path string** straight into the `{"type":"video",...}` block (matching
`eval_videophy2.py`) rather than pre-decoding frames ourselves. The optional
`video_*` knobs above tune that sampling.

## Validation & error handling (fail fast, clear messages)

- **Image + video together** — rejected at `_impl_generate_reasoner_text` and at
  `OmniMoTModel.generate_reasoner_text`. The reasoner conditions on one medium at
  a time.
- **Video pairing** — `pixel_values_videos` and `video_grid_thw` must both be
  present or both absent (mirrors the existing image-pair guard).
- **`video_fps` + `video_num_frames` together** — rejected in the schema,
  mirroring the processor's own mutual-exclusion rule.
- **Batch homogeneity** — extend the current "no mixing image-conditioned and
  text-only" check in `_generate_reasoner_batch` to three kinds: a batch is
  all-text, all-image, or all-video. Mixed -> `ValueError` telling the user to
  split inputs.
- **No vision tower** — already handled: `_impl` raises if `causal_lm.visual` is
  missing.
- **Placeholder-token mismatch** — already handled: `get_placeholder_mask`
  raises if the video token count != produced features.
- **Extension routing** — relies on the existing `VIDEO_EXTENSIONS` /
  `IMAGE_EXTENSIONS` sets in `args.py`; an unrecognized extension already raises
  `Invalid vision extension`.

## Non-goals / notes

- **Mixed image+video in one prompt** — out of scope.
- **Input-video content-safety guardrail** — none today; not added. The reasoner
  emits only text, never video, so the text guardrail on prompt and output is
  unchanged and sufficient.
- **Video decode backend** — the processor needs a video backend
  (decord / torchvision) to read the mp4; if missing, the failure surfaces inside
  `apply_chat_template`. We do not add our own decode path. This is an
  environment dependency to document, not code we write.
- **Unused output vision fields** — `fps` / `num_frames` / resolution remain
  unused by the reasoner (already defaulted in `args.py`).

## Verification (manual only)

No automated test for now. The implementation ships the artifacts to verify by
hand.

Example input `inputs/reasoner/reasoner_video.json`:

```json
{
    "model_mode": "reasoner",
    "prompt": "Describe what happens in this video in one sentence.",
    "vision_path": "/abs/path/to/clip.mp4",
    "video_fps": 2,
    "video_max_pixels": 200704
}
```

(`video_*` fields optional — omit to use processor defaults.)

Run (Nano; Super identical but `--checkpoint-path Cosmos3-Super`):

```bash
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput --dp-shard-size=8 --dp-replicate-size=1 \
    --cp-size=1 --cfgp-size=1 \
    -i "inputs/reasoner/reasoner_video.json" \
    -o outputs/reasoner_video --checkpoint-path Cosmos3-Nano --seed=0
```

Expected: `outputs/reasoner_video/reasoner_video/reasoner_text.txt` contains
non-empty, on-topic text describing the clip; no crash; image and text-only
reasoner inputs still work unchanged.

A parity check against the HF `eval_videophy2.py` path is a possible future
hardening step, out of scope here.

## Files touched

| File | Change |
| ---- | ------ |
| `cosmos_framework/model/vfm/vlm/qwen3_vl/utils.py` | `prepare_multimodal_reasoner_inputs`: add video branch |
| `cosmos_framework/model/vfm/mot/unified_mot.py` | `_impl_generate_reasoner_text` + wrapper `generate_reasoner_text`: add/forward video params |
| `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py` | `generate_reasoner_text`: forward video params |
| `cosmos_framework/model/vfm/omni_mot_model.py` | `generate_reasoner_text`: `videos` param, video chat block, sampling kwargs |
| `cosmos_framework/inference/inference.py` | `_get_reasoner_sample_data` + `_generate_reasoner_batch`: route mp4 |
| `cosmos_framework/inference/args.py` | add `video_*` sampling fields + mutual-exclusion validation |
| `cosmos_framework/inference/defaults/reasoner/sample_args.json` | add `video_*` defaults (`null`) |
| `inputs/reasoner/reasoner_video.json` | new example input |
| `docs/inference.md` | document video input + `video_*` fields for `reasoner` mode |
```