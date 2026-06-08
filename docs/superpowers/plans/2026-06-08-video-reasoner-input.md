# Video input for `reasoner` model-mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `model_mode=reasoner` in `python -m cosmos_framework.scripts.inference` accept a local mp4 video as conditioning input (Cosmos3-Nano and Cosmos3-Super), producing text that reasons over the clip.

**Architecture:** Additive "video lane" alongside the existing image lane through the reasoner wrapper stack. The vendored Qwen3-VL model + `video_processing_qwen3_vl.py` already implement video end to end (`get_video_features`, `get_rope_index(video_grid_thw=…)`, `get_placeholder_mask(video_features=…)`, `video_token_id`); only the Cosmos wrapper layers are hardcoded to images. We thread optional `pixel_values_videos` / `video_grid_thw` (and a high-level `videos` list + `video_*` sampling knobs) through five layers. A prompt carries either an image, a video, or neither — never both.

**Tech Stack:** Python, PyTorch, pydantic, HuggingFace transformers (`Qwen3VLProcessor`), torchrun. Repo lives at `cosmos-framework/`; spec at `docs/superpowers/specs/2026-06-07-video-reasoner-input-design.md`.

**Verification policy (read before starting):** Per the spec, the **end-to-end video path is verified manually on GPU** (Task 9) — there is no automated GPU test, because it requires real checkpoints + multi-GPU. The two pure-Python/logic changes (args schema in Task 1, builder routing in Task 7) DO get real `pytest` unit tests (CPU-only, no checkpoints). Model-layer tasks (3–6) are verified by import/lint checks plus the Task 9 manual run.

**How to run tests/commands:** Python/pytest must run inside the i4 container (`bob_echo_dev`). Use the `cosmos3-run-env` skill to author the wrapper shell and the `slurm-node` skill to execute. Where a step says `pytest …`, it means "run that inside the container."

---

## File Structure

| File | Responsibility | Change |
| ---- | -------------- | ------ |
| `cosmos_framework/inference/args.py` | Sample-arg schema | Add `video_*` reasoner fields + mutual-exclusion validation |
| `cosmos_framework/inference/args_test.py` | Schema unit tests | Add tests for new fields/validation |
| `cosmos_framework/inference/defaults/reasoner/sample_args.json` | Reasoner defaults | Add `video_*` keys (null) |
| `cosmos_framework/model/vfm/vlm/qwen3_vl/utils.py` | Multimodal prefill | `prepare_multimodal_reasoner_inputs`: add video branch |
| `cosmos_framework/model/vfm/mot/unified_mot.py` | Reasoner decode | `_impl_generate_reasoner_text` + 3 wrapper `generate_reasoner_text`: add/forward video params + guards |
| `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py` | Network pass-through | `generate_reasoner_text`: forward video params |
| `cosmos_framework/model/vfm/omni_mot_model.py` | High-level entry | `generate_reasoner_text`: `videos` param, video chat block, sampling kwargs |
| `cosmos_framework/inference/inference.py` | Inference engine | `_get_reasoner_sample_data` route mp4; `_generate_reasoner_batch` homogeneity + video forward |
| `cosmos_framework/inference/inference_test.py` | Builder unit test | Add routing test (CPU) |
| `inputs/reasoner/reasoner_video.json` | Example input | New file |
| `docs/inference.md` | User docs | Document video input + `video_*` fields |

Implementation order: Task 1 (schema) → Task 2 (defaults) → Tasks 3–6 (model layers, bottom-up) → Task 7 (inference wiring) → Task 8 (docs/example) → Task 9 (manual GPU verification).

---

## Task 1: Add `video_*` reasoner sample-arg fields + validation

**Files:**
- Modify: `cosmos_framework/inference/args.py` (class `ReasonerDataArgs` ~600-611, class `ReasonerDataOverrides` ~614-638)
- Test: `cosmos_framework/inference/args_test.py`

The new fields control how the Qwen3-VL processor samples frames from the mp4. They are `video_`-prefixed to avoid colliding with the existing output-oriented `fps`/`num_frames` fields. `video_fps` and `video_num_frames` are mutually exclusive (the processor itself raises if both are set).

- [ ] **Step 1: Write the failing tests**

Add to `cosmos_framework/inference/args_test.py` (match the existing test style/imports in that file; these construct a reasoner override and resolve it). If the file already has a helper to build an `OmniSampleOverrides`/model config, reuse it; otherwise mirror the nearest existing reasoner test.

```python
def test_reasoner_video_fields_default_none():
    ov = ReasonerDataOverrides()
    assert ov.video_fps is None
    assert ov.video_num_frames is None
    assert ov.video_min_frames is None
    assert ov.video_max_frames is None
    assert ov.video_min_pixels is None
    assert ov.video_max_pixels is None


def test_reasoner_video_fps_and_num_frames_mutually_exclusive():
    import pytest
    ov = ReasonerDataOverrides(video_fps=2, video_num_frames=16)
    # _validate_video_sampling is called from _build_reasoner_data; call it directly
    with pytest.raises(ValueError, match="video_fps.*video_num_frames|mutually exclusive"):
        ov._validate_video_sampling()


def test_reasoner_video_fps_alone_ok():
    ov = ReasonerDataOverrides(video_fps=2)
    ov._validate_video_sampling()  # no raise
```

Add the import for `ReasonerDataOverrides` to the test file's import block if not present:

```python
from cosmos_framework.inference.args import ReasonerDataOverrides
```

- [ ] **Step 2: Run tests to verify they fail**

Run (inside container): `pytest cosmos_framework/inference/args_test.py -k reasoner_video -v`
Expected: FAIL — `ReasonerDataOverrides` has no `video_fps` (AttributeError / unexpected-keyword), and no `_validate_video_sampling`.

- [ ] **Step 3: Add the fields to `ReasonerDataArgs`**

In `args.py`, append to class `ReasonerDataArgs` (after `presence_penalty: float | None = None`, ~line 611):

```python
    video_fps: float | None = None
    video_num_frames: pydantic.PositiveInt | None = None
    video_min_frames: pydantic.PositiveInt | None = None
    video_max_frames: pydantic.PositiveInt | None = None
    video_min_pixels: pydantic.PositiveInt | None = None
    video_max_pixels: pydantic.PositiveInt | None = None
```

- [ ] **Step 4: Add the fields + validation to `ReasonerDataOverrides`**

In `args.py`, append to class `ReasonerDataOverrides` (after `presence_penalty`, ~line 631, before `_build_reasoner_data`):

```python
    video_fps: float | None = None
    """Frames per second to sample from a video vision_path. Mutually exclusive with video_num_frames. None -> processor default."""
    video_num_frames: pydantic.PositiveInt | None = None
    """Fixed number of frames to sample from a video vision_path. Mutually exclusive with video_fps. None -> processor default."""
    video_min_frames: pydantic.PositiveInt | None = None
    """Lower bound on sampled frame count. None -> processor default."""
    video_max_frames: pydantic.PositiveInt | None = None
    """Upper bound on sampled frame count. None -> processor default."""
    video_min_pixels: pydantic.PositiveInt | None = None
    """Lower bound on per-frame pixel budget (drives smart_resize). None -> processor default."""
    video_max_pixels: pydantic.PositiveInt | None = None
    """Upper bound on per-frame pixel budget (drives smart_resize). None -> processor default."""

    def _validate_video_sampling(self) -> None:
        if self.video_fps is not None and self.video_num_frames is not None:
            raise ValueError(
                "video_fps and video_num_frames are mutually exclusive — set at most one."
            )
```

Then call it from `_build_reasoner_data` so resolution-time validation fires. Replace the body of `_build_reasoner_data` (~lines 633-638) with:

```python
    def _build_reasoner_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        if not sample_meta.model_mode.is_reasoner:
            return
        self = cast("SampleDataOverrides", self)
        if not self.prompt.strip():
            raise ValueError("Reasoner inference requires a non-empty 'prompt'.")
        self._validate_video_sampling()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest cosmos_framework/inference/args_test.py -k reasoner_video -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Lint + commit**

```bash
ruff check cosmos_framework/inference/args.py cosmos_framework/inference/args_test.py
git add cosmos_framework/inference/args.py cosmos_framework/inference/args_test.py
git commit -m "feat(reasoner): add video_* sampling fields + mutual-exclusion validation"
```

---

## Task 2: Add `video_*` defaults to the reasoner defaults file

**Files:**
- Modify: `cosmos_framework/inference/defaults/reasoner/sample_args.json`

`None` defaults already live in the schema; adding explicit `null` keys here documents the knobs and keeps the defaults file self-describing.

- [ ] **Step 1: Edit the JSON**

Replace the file contents with:

```json
{
    "model_mode": "reasoner",
    "max_new_tokens": 64,
    "do_sample": false,
    "temperature": 1.0,
    "top_k": null,
    "top_p": null,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "video_fps": null,
    "video_num_frames": null,
    "video_min_frames": null,
    "video_max_frames": null,
    "video_min_pixels": null,
    "video_max_pixels": null
}
```

- [ ] **Step 2: Verify it loads**

Run (inside container):
`python -c "from cosmos_framework.inference.args import _load_modality_defaults; print(_load_modality_defaults('reasoner'))"`
Expected: prints the dict including the `video_*` keys; no exception.

- [ ] **Step 3: Commit**

```bash
git add cosmos_framework/inference/defaults/reasoner/sample_args.json
git commit -m "feat(reasoner): add video_* defaults (null) to reasoner sample_args"
```

---

## Task 3: Add a video branch to `prepare_multimodal_reasoner_inputs`

**Files:**
- Modify: `cosmos_framework/model/vfm/vlm/qwen3_vl/utils.py:497-604`

This is the one real seam. The image recipe (lines 577-604) is: `get_image_features` → `get_placeholder_mask(image_features=…)` → `masked_scatter(image_mask)` → `get_rope_index(image_grid_thw=…)`. The video recipe is identical but uses the video helpers — and `get_video_features` is literally "same implementation as for images" (`qwen3_vl.py:1243`), so we reuse the existing free `get_image_features` helper with the video tensors. `get_placeholder_mask` and `get_rope_index` already accept video arguments.

- [ ] **Step 1: Add optional video params to the signature**

Change the signature (lines 497-509) to add two params after `image_grid_thw`:

```python
def prepare_multimodal_reasoner_inputs(
    causal_lm: Any,
    input_ids: torch.Tensor,  # [B,T_prompt]
    pixel_values: torch.Tensor | None = None,  # [N_patches,C,H,W]
    image_grid_thw: torch.Tensor | None = None,  # [num_images,3]
    pixel_values_videos: torch.Tensor | None = None,  # [N_patches,C,H,W]
    video_grid_thw: torch.Tensor | None = None,  # [num_videos,3]
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,  # inputs_embeds [B,T_prompt,hidden_size]
    torch.Tensor,  # visual_pos_masks [B,T_prompt] bool
    list[torch.Tensor],  # deepstack_visual_embeds (per deepstack layer)
    torch.Tensor,  # position_ids
    torch.Tensor,  # mrope_position_deltas
]:
```

(Note: `pixel_values`/`image_grid_thw` are now defaulted to `None`; existing callers pass them positionally/by keyword so behavior is unchanged.)

- [ ] **Step 2: Replace the body (lines 577-604) with image/video branching**

```python
    is_video = pixel_values_videos is not None
    inputs_embeds = causal_lm.model.embed_tokens(input_ids).clone()  # [B,T_prompt,hidden_size]

    if is_video:
        pixel_values_videos = pixel_values_videos.to(device=inputs_embeds.device)
        video_grid_thw = video_grid_thw.to(device=inputs_embeds.device)
        # get_video_features == get_image_features (same visual tower); reuse the free helper.
        video_embeds, deepstack_visual_embeds = get_image_features(causal_lm, pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        _image_mask, video_mask = get_placeholder_mask(
            causal_lm,
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)  # [B,T_prompt,hidden_size]
        visual_pos_masks = video_mask[..., 0]  # [B,T_prompt]
    else:
        pixel_values = pixel_values.to(device=inputs_embeds.device)
        image_grid_thw = image_grid_thw.to(device=inputs_embeds.device)
        image_embeds, deepstack_visual_embeds = get_image_features(causal_lm, pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_mask, _video_mask = get_placeholder_mask(
            causal_lm,
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # [B,T_prompt,hidden_size]
        visual_pos_masks = image_mask[..., 0]  # [B,T_prompt]

    deepstack_visual_embeds = [
        embed.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype) for embed in deepstack_visual_embeds
    ]

    position_ids, mrope_position_deltas = get_rope_index(
        causal_lm,
        input_ids=input_ids,
        image_grid_thw=None if is_video else image_grid_thw,
        video_grid_thw=video_grid_thw if is_video else None,
        attention_mask=attention_mask,
    )

    return inputs_embeds, visual_pos_masks, deepstack_visual_embeds, position_ids, mrope_position_deltas
```

- [ ] **Step 3: Update the docstring**

In the docstring (lines 528-532), replace the sentence "Videos and dual image+video paths are not supported here; only `image_grid_thw` is consumed…" with:

```
    Either the image pair (``pixel_values`` + ``image_grid_thw``) or the
    video pair (``pixel_values_videos`` + ``video_grid_thw``) is consumed —
    not both. The video recipe mirrors the image recipe but routes through
    the video placeholder mask and ``video_grid_thw`` rope index.
```

- [ ] **Step 4: Import/lint check (no GPU test — verified end-to-end in Task 9)**

Run (inside container):
`python -c "import cosmos_framework.model.vfm.vlm.qwen3_vl.utils"`
Expected: no ImportError / SyntaxError.
`ruff check cosmos_framework/model/vfm/vlm/qwen3_vl/utils.py`

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/model/vfm/vlm/qwen3_vl/utils.py
git commit -m "feat(reasoner): video branch in prepare_multimodal_reasoner_inputs"
```

---

## Task 4: Thread video params through `_impl_generate_reasoner_text`

**Files:**
- Modify: `cosmos_framework/model/vfm/mot/unified_mot.py:1490-1675`

- [ ] **Step 1: Add params to the signature**

In `_impl_generate_reasoner_text` (lines 1490-1508), add two params after `image_grid_thw` (line 1496):

```python
    pixel_values_videos: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
```

- [ ] **Step 2: Extend the validation guard**

Replace the guard at lines 1644-1645:

```python
    if (pixel_values is None) != (image_grid_thw is None):
        raise ValueError("pixel_values and image_grid_thw must be provided together.")
```

with:

```python
    if (pixel_values is None) != (image_grid_thw is None):
        raise ValueError("pixel_values and image_grid_thw must be provided together.")
    if (pixel_values_videos is None) != (video_grid_thw is None):
        raise ValueError("pixel_values_videos and video_grid_thw must be provided together.")
    if pixel_values is not None and pixel_values_videos is not None:
        raise ValueError("Reasoner conditions on one medium at a time: pass image OR video, not both.")
```

- [ ] **Step 3: Route to the prefill helper for both media**

Replace the prefill branch at lines 1650-1667:

```python
    if pixel_values is None:
        hidden = model.reasoner_forward(input_ids, cache=cache)  # [B,T_prompt,hidden_size]
    else:
        if not hasattr(causal_lm, "visual"):
            raise ValueError("Combined checkpoint does not include a visual module on the reasoner language model.")
        (
            inputs_embeds,
            visual_pos_masks,
            deepstack_visual_embeds,
            position_ids,
            mrope_position_deltas,
        ) = prepare_multimodal_reasoner_inputs(
            causal_lm,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
```

with:

```python
    if pixel_values is None and pixel_values_videos is None:
        hidden = model.reasoner_forward(input_ids, cache=cache)  # [B,T_prompt,hidden_size]
    else:
        if not hasattr(causal_lm, "visual"):
            raise ValueError("Combined checkpoint does not include a visual module on the reasoner language model.")
        (
            inputs_embeds,
            visual_pos_masks,
            deepstack_visual_embeds,
            position_ids,
            mrope_position_deltas,
        ) = prepare_multimodal_reasoner_inputs(
            causal_lm,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
```

- [ ] **Step 4: Update the docstring**

In the `pixel_values` docstring (lines 1553-1556), replace "Videos are *not* supported here — this function has no `pixel_values_videos` / `video_grid_thw` parameters; for I2V conditioning, frames must be passed as images." with:

```
            For video conditioning, pass ``pixel_values_videos`` +
            ``video_grid_thw`` instead (mutually exclusive with the image
            pair).
```

- [ ] **Step 5: Import/lint check**

Run (inside container):
`python -c "import cosmos_framework.model.vfm.mot.unified_mot"`
`ruff check cosmos_framework/model/vfm/mot/unified_mot.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add cosmos_framework/model/vfm/mot/unified_mot.py
git commit -m "feat(reasoner): accept video tensors in _impl_generate_reasoner_text"
```

---

## Task 5: Forward video params through the wrapper `generate_reasoner_text` pass-throughs

**Files:**
- Modify: `cosmos_framework/model/vfm/mot/unified_mot.py` — three wrappers at lines 1932 (`Qwen3VLTextForCausalLM`), 2060 (`Qwen3VLMoeTextForCausalLM`), 2184 (`Nemotron3DenseVLTextForCausalLM`)
- Modify: `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py:272-341`

All four are pure pass-throughs to `_impl_generate_reasoner_text` (the three unified_mot wrappers) and to `self.language_model.generate_reasoner_text` (the network). Each needs the two new params added to its signature and forwarded.

- [ ] **Step 1: Update the three unified_mot wrappers**

For EACH of the three `generate_reasoner_text` methods (lines 1932, 2060, 2184): add after `image_grid_thw: torch.Tensor | None = None,` in the signature:

```python
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
```

and add to the `_impl_generate_reasoner_text(...)` call (after `image_grid_thw=image_grid_thw,`):

```python
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
```

(The three methods are textually identical in this region; apply the same two-line additions to each.)

- [ ] **Step 2: Update the network pass-through**

In `cosmos3_vfm_network.py`, add to the `generate_reasoner_text` signature (after `image_grid_thw: torch.Tensor | None = None,`, ~line 278):

```python
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
```

and to the forwarded call (after `image_grid_thw=image_grid_thw,`, ~line 329):

```python
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
```

- [ ] **Step 3: Import/lint check**

Run (inside container):
`python -c "import cosmos_framework.model.vfm.mot.unified_mot, cosmos_framework.model.vfm.mot.cosmos3_vfm_network"`
`ruff check cosmos_framework/model/vfm/mot/unified_mot.py cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add cosmos_framework/model/vfm/mot/unified_mot.py cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py
git commit -m "feat(reasoner): forward video tensors through generate_reasoner_text pass-throughs"
```

---

## Task 6: Add `videos` + sampling kwargs to `OmniMoTModel.generate_reasoner_text`

**Files:**
- Modify: `cosmos_framework/model/vfm/omni_mot_model.py:3760-4007`

This builds a `{"type":"video", ...}` chat block (parallel to the existing image block at lines 3959-4008), extracts `pixel_values_videos` / `video_grid_thw` from `apply_chat_template`, and passes them down.

- [ ] **Step 1: Add params to the signature**

In `generate_reasoner_text` (lines 3760-3774), add after `images: list[Any] | None = None,` (line 3765):

```python
        videos: list[Any] | None = None,
        video_sampling_kwargs: dict[str, Any] | None = None,
```

- [ ] **Step 2: Validate not-both and set the multimodal flag**

Replace the validation block at lines 3907-3922 (`use_multimodal = images is not None` … through the `apply_chat_template` RuntimeError) with:

```python
        if images is not None and videos is not None:
            raise ValueError("generate_reasoner_text conditions on one medium at a time: pass `images` OR `videos`, not both.")
        use_image = images is not None
        use_video = videos is not None
        use_multimodal = use_image or use_video
        media = images if use_image else videos
        if use_multimodal:
            assert media is not None  # narrowed by `use_multimodal`
            if len(media) != len(inputs):
                raise ValueError(
                    f"generate_reasoner_text: media length ({len(media)}) "
                    f"must equal `inputs` length ({len(inputs)}) for the "
                    "vision-conditioned flow."
                )
            if not callable(getattr(self.vlm_processor, "apply_chat_template", None)):
                raise RuntimeError(
                    "generate_reasoner_text(images=/videos=...) requires a multimodal "
                    "VLM processor (e.g. Qwen3VLProcessor) but the live processor "
                    f"{type(self.vlm_processor).__name__!r} does not implement "
                    "apply_chat_template — the live VLM is configured as text-only."
                )
        video_kwargs = {k: v for k, v in (video_sampling_kwargs or {}).items() if v is not None}
```

- [ ] **Step 3: Build the image-or-video chat block and extract tensors**

Replace the multimodal block construction at lines 3959-4008 (`if use_multimodal:` … through the `out_ids = self.net.generate_reasoner_text(...)` image call) with:

```python
            if use_multimodal:
                assert media is not None  # narrowed by `use_multimodal`
                # Replace the LAST user message's content with a Qwen3-VL
                # multimodal block. Earlier messages (system, prior turns)
                # are kept verbatim.
                last_user = messages[-1]
                last_text = last_user["content"] if isinstance(last_user.get("content"), str) else ""
                if use_video:
                    media_item: dict[str, Any] = {"type": "video", "video": media[idx]}
                else:
                    media_item = {"type": "image", "image": media[idx]}
                multimodal_messages = list(messages[:-1])
                multimodal_messages.append(
                    {
                        "role": "user",
                        "content": [media_item, {"type": "text", "text": last_text}],
                    }
                )
                # NOTE: `video_kwargs` (fps/num_frames/min_frames/max_frames/
                # min_pixels/max_pixels) are forwarded to the processor here.
                # The exact kwarg surface depends on the installed transformers
                # Qwen3VLProcessor; if a key is rejected, route via the
                # processor's video-loading kwargs. Verified manually in the
                # plan's Task 9.
                processor_inputs = self.vlm_processor.apply_chat_template(
                    multimodal_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    **(video_kwargs if use_video else {}),
                )
                inner_input_ids = processor_inputs["input_ids"].to(device).unsqueeze(0)
                inner_attention_mask = processor_inputs["attention_mask"].to(device).unsqueeze(0)
                if use_video:
                    inner_pixel_values_videos = processor_inputs["pixel_values_videos"].to(device)
                    inner_video_grid_thw = processor_inputs["video_grid_thw"].to(device)
                    out_ids = self.net.generate_reasoner_text(
                        input_ids=inner_input_ids,
                        max_new_tokens=max_new_tokens,
                        pixel_values_videos=inner_pixel_values_videos,
                        video_grid_thw=inner_video_grid_thw,
                        attention_mask=inner_attention_mask,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        do_sample=do_sample,
                        temperature=temperature if temperature is not None else 1.0,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        presence_penalty=presence_penalty,
                        seed=seed,
                        return_only_new_tokens=True,
                    )
                else:
                    inner_pixel_values = processor_inputs["pixel_values"].to(device)  # [N_patches,C,H,W]
                    inner_image_grid_thw = processor_inputs["image_grid_thw"].to(device)  # [num_images,3]
                    out_ids = self.net.generate_reasoner_text(
                        input_ids=inner_input_ids,
                        max_new_tokens=max_new_tokens,
                        pixel_values=inner_pixel_values,
                        image_grid_thw=inner_image_grid_thw,
                        attention_mask=inner_attention_mask,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        do_sample=do_sample,
                        temperature=temperature if temperature is not None else 1.0,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        presence_penalty=presence_penalty,
                        seed=seed,
                        return_only_new_tokens=True,
                    )
```

(The text-only `else:` branch at lines 4009+ is unchanged.)

- [ ] **Step 4: Update the docstring**

In the `images:` Args entry (~lines 3828-3837), add a sibling paragraph:

```
            videos: Optional per-prompt conditioning videos (mutually
                exclusive with ``images``). Each entry is forwarded into a
                ``{"type": "video", "video": ...}`` chat block; the
                processor decodes/samples frames and produces
                ``pixel_values_videos`` / ``video_grid_thw``.
            video_sampling_kwargs: Optional dict of non-None frame-sampling
                controls (fps, num_frames, min_frames, max_frames,
                min_pixels, max_pixels) forwarded to the processor.
```

- [ ] **Step 5: Import/lint check**

Run (inside container):
`python -c "import cosmos_framework.model.vfm.omni_mot_model"`
`ruff check cosmos_framework/model/vfm/omni_mot_model.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add cosmos_framework/model/vfm/omni_mot_model.py
git commit -m "feat(reasoner): videos param + video chat block in OmniMoTModel.generate_reasoner_text"
```

---

## Task 7: Wire mp4 routing into the inference engine

**Files:**
- Modify: `cosmos_framework/inference/inference.py` — `_get_reasoner_sample_data:466-474`, `_generate_reasoner_batch:1644-1696`
- Test: `cosmos_framework/inference/inference_test.py`

The builder detects an mp4 `vision_path` by extension and returns it under a `reasoner_videos` key (path string, not decoded) plus the resolved `video_*` sampling kwargs; the batch method routes videos to `generate_reasoner_text(videos=…)`.

- [ ] **Step 1: Write the failing routing test**

Add to `cosmos_framework/inference/inference_test.py` (use `types.SimpleNamespace` to avoid constructing a full model/args; the builder only reads `vision_path`, `prompt`, and `video_*` off `sample_args`, and `input_caption_key` off `model`):

```python
import types
from cosmos_framework.inference.inference import _get_reasoner_sample_data


def _fake_sa(vision_path, **video_kw):
    base = dict(
        prompt="describe",
        vision_path=vision_path,
        video_fps=None, video_num_frames=None, video_min_frames=None,
        video_max_frames=None, video_min_pixels=None, video_max_pixels=None,
    )
    base.update(video_kw)
    return types.SimpleNamespace(**base)


_fake_model = types.SimpleNamespace(input_caption_key="caption")


def test_reasoner_sample_data_text_only():
    out = _get_reasoner_sample_data(_fake_sa(None), _fake_model)
    assert out["caption"] == ["describe"]
    assert out["reasoner_images"] == [None]
    assert "reasoner_videos" not in out


def test_reasoner_sample_data_video_routes_to_videos(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")  # not decoded by the builder
    out = _get_reasoner_sample_data(_fake_sa(str(clip), video_fps=2), _fake_model)
    assert out["caption"] == ["describe"]
    assert out["reasoner_videos"] == [str(clip)]
    assert out["reasoner_images"] == [None]
    assert out["video_sampling_kwargs"] == {"fps": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run (inside container): `pytest cosmos_framework/inference/inference_test.py -k reasoner_sample_data -v`
Expected: FAIL — current builder always calls `Image.open` and has no `reasoner_videos`/`video_sampling_kwargs` keys.

- [ ] **Step 3: Add the `VIDEO_EXTENSIONS` import**

`VIDEO_EXTENSIONS` is exported from `cosmos_framework.inference.common.args` (the same module `args.py` imports it from). `inference.py` already imports `Path`, `Any`, `cast`, and `Image`, so this is the only new import. Add near the top of `inference.py`:

```python
from cosmos_framework.inference.common.args import VIDEO_EXTENSIONS
```

(If `inference.py` already imports other names from `cosmos_framework.inference.common.args`, append `VIDEO_EXTENSIONS` to that existing import instead of adding a new line.)

- [ ] **Step 4: Rewrite `_get_reasoner_sample_data`**

Replace lines 466-474:

```python
def _get_reasoner_sample_data(sample_args: OmniSampleArgs, model: OmniMoTModel) -> dict[str, Any]:
    """Sample batch for reasoner text generation: prompt + optional conditioning image or video."""
    image: Image.Image | None = None
    video: str | None = None
    if sample_args.vision_path is not None:
        if Path(sample_args.vision_path).suffix.lower() in VIDEO_EXTENSIONS:
            video = str(sample_args.vision_path)
        else:
            image = Image.open(sample_args.vision_path).convert("RGB")
    out: dict[str, Any] = {
        model.input_caption_key: [sample_args.prompt],
        "reasoner_images": [image],
    }
    if video is not None:
        out["reasoner_videos"] = [video]
        out["video_sampling_kwargs"] = {
            k: v
            for k, v in {
                "fps": sample_args.video_fps,
                "num_frames": sample_args.video_num_frames,
                "min_frames": sample_args.video_min_frames,
                "max_frames": sample_args.video_max_frames,
                "min_pixels": sample_args.video_min_pixels,
                "max_pixels": sample_args.video_max_pixels,
            }.items()
            if v is not None
        }
    return out
```

- [ ] **Step 5: Run the routing tests**

Run: `pytest cosmos_framework/inference/inference_test.py -k reasoner_sample_data -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Update `_generate_reasoner_batch` to route videos**

In `_generate_reasoner_batch` (lines 1656-1696), after `raw_images: list[...] = data_batch["reasoner_images"]` (line 1657), add video extraction and a three-way homogeneity check, then branch the model call. Replace lines 1656-1696 (`prompts = ...` through the `generate_reasoner_text(...)` call) with:

```python
        prompts: list[str] = data_batch[self.model.input_caption_key]
        raw_images: list[Image.Image | None] = data_batch["reasoner_images"]
        raw_videos: list[str | None] | None = data_batch.get("reasoner_videos")
        video_sampling_kwargs: dict[str, Any] = data_batch.get("video_sampling_kwargs", {})

        n_img = sum(img is not None for img in raw_images)
        n_vid = sum(v is not None for v in (raw_videos or []))
        if n_img and n_vid:
            raise ValueError(
                "Reasoner batch mixes image- and video-conditioned samples. Split into separate batches."
            )
        if 0 < n_img < len(raw_images):
            raise ValueError(
                "Reasoner batch mixes image-conditioned and text-only samples "
                f"({n_img}/{len(raw_images)} have an image vision_path). Split into separate batches."
            )
        if raw_videos is not None and 0 < n_vid < len(raw_videos):
            raise ValueError(
                "Reasoner batch mixes video-conditioned and text-only samples "
                f"({n_vid}/{len(raw_videos)} have a video vision_path). Split into separate batches."
            )
        images: list[Image.Image] | None = cast(list[Image.Image], raw_images) if n_img == len(raw_images) else None
        videos: list[str] | None = (
            cast(list[str], raw_videos) if raw_videos is not None and n_vid == len(raw_videos) else None
        )

        try:
            with sync_distributed_errors():
                for sa, prompt in zip(sample_args_list, prompts):
                    if self.should_process_sample(sa) and not warmup:
                        log.debug(f"{sa.__class__.__name__}({sa})")
                        assert sa.output_dir is not None
                        sa.output_dir.mkdir(parents=True, exist_ok=True)
                        (sa.output_dir / "sample_args.json").write_text(sa.model_dump_json())
                        self._run_text_guardrail(str(sa.output_dir), prompt)
        except Exception as e:
            return [
                self._handle_sample_exception(sa, e)
                for sa in sample_args_list
                if self.should_process_sample(sa) and not warmup
            ]

        with self._get_timer(f"{self.model.__class__.__name__}.generate_reasoner_text"):
            texts = self.model.generate_reasoner_text(
                prompts,
                max_new_tokens=sample_args_list[0].max_new_tokens,
                images=images,
                videos=videos,
                video_sampling_kwargs=video_sampling_kwargs or None,
                do_sample=sample_args_list[0].do_sample,
                temperature=sample_args_list[0].temperature,
                top_k=sample_args_list[0].top_k,
                top_p=sample_args_list[0].top_p,
                repetition_penalty=sample_args_list[0].repetition_penalty,
                presence_penalty=sample_args_list[0].presence_penalty,
                seed=sample_args_list[0].seed,
            )
```

(Confirm `Any` and `cast` are already imported in `inference.py`; both are used elsewhere in the file, so no new import is needed.)

- [ ] **Step 7: Import/lint check + run builder tests again**

Run (inside container):
`python -c "import cosmos_framework.inference.inference"`
`ruff check cosmos_framework/inference/inference.py cosmos_framework/inference/inference_test.py`
`pytest cosmos_framework/inference/inference_test.py -k reasoner_sample_data -v`
Expected: no errors; tests PASS.

- [ ] **Step 8: Commit**

```bash
git add cosmos_framework/inference/inference.py cosmos_framework/inference/inference_test.py
git commit -m "feat(reasoner): route mp4 vision_path to video conditioning in inference engine"
```

---

## Task 8: Example input + user docs

**Files:**
- Create: `inputs/reasoner/reasoner_video.json`
- Modify: `docs/inference.md`

- [ ] **Step 1: Create the example input**

`inputs/reasoner/reasoner_video.json`:

```json
{
    "model_mode": "reasoner",
    "prompt": "Describe what happens in this video in one sentence.",
    "vision_path": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_153.jpg"
}
```

NOTE: replace the placeholder `vision_path` with a real `.mp4` URL or local path before running. If a canonical sample mp4 exists under the cosmos-dependencies repo, use that; otherwise leave a local-path example and document it. (Confirm a sample clip during Task 9; update this file to point at it.)

- [ ] **Step 2: Document in `docs/inference.md`**

In the Modes table (around line 138-146), the reasoner mode is currently text/image only. Add a row or note documenting video input for `reasoner`. Find the reasoner documentation block and add:

```markdown
For `model_mode=reasoner`, `vision_path` may point to an **image** (`.jpg`/`.png`/…) or a **video** (`.mp4`/…). A video is decoded by the Qwen3-VL processor and sampled into frames. Optional frame-sampling controls (all default to the processor's defaults):

- `video_fps`: frames sampled per second (mutually exclusive with `video_num_frames`).
- `video_num_frames`: fixed number of frames to sample.
- `video_min_frames` / `video_max_frames`: bounds on the sampled frame count.
- `video_min_pixels` / `video_max_pixels`: per-frame pixel budget (drives resolution).

Example: [`inputs/reasoner/reasoner_video.json`](../inputs/reasoner/reasoner_video.json).
```

- [ ] **Step 3: Verify the example JSON parses**

Run: `python -c "import json; json.load(open('inputs/reasoner/reasoner_video.json'))"`
Expected: no exception.

- [ ] **Step 4: Commit**

```bash
git add inputs/reasoner/reasoner_video.json docs/inference.md
git commit -m "docs(reasoner): document video input + add reasoner_video example"
```

---

## Task 9: Manual end-to-end GPU verification

**Files:** none (verification only). Use the `cosmos3-run-env` skill to author the wrapper and `slurm-node` to run on a GPU node in the i4 container.

This is the real correctness gate (per the spec: manual verification only). Do NOT mark the feature complete until this passes.

- [ ] **Step 1: Obtain a short sample mp4**

Place a short clip at a known path, e.g. `tmp_inputs/clip.mp4` (a few seconds is enough). Update `inputs/reasoner/reasoner_video.json`'s `vision_path` to that absolute path (or a real mp4 URL).

- [ ] **Step 2: Run reasoner video inference on Cosmos3-Nano**

```bash
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput --dp-shard-size=8 --dp-replicate-size=1 \
    --cp-size=1 --cfgp-size=1 \
    -i "inputs/reasoner/reasoner_video.json" \
    -o outputs/reasoner_video --checkpoint-path Cosmos3-Nano --seed=0
```

Expected: completes without error; `outputs/reasoner_video/reasoner_video/reasoner_text.txt` exists and contains non-empty, on-topic text describing the clip.

- [ ] **Step 3: Repeat for Cosmos3-Super**

Same command with `--checkpoint-path Cosmos3-Super`. Expected: same success criteria.

- [ ] **Step 4: Regression — confirm image and text-only reasoner still work**

```bash
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput --dp-shard-size=8 --dp-replicate-size=1 \
    --cp-size=1 --cfgp-size=1 \
    -i "inputs/reasoner/reasoner.json" -i "inputs/reasoner/reasoner_image.json" \
    -o outputs/reasoner_regress --checkpoint-path Cosmos3-Nano --seed=0
```

Expected: both produce `reasoner_text.txt` with non-empty text, unchanged from pre-change behavior.

- [ ] **Step 5: Sampling-knob smoke check**

Add `"video_fps": 1` (then separately `"video_num_frames": 8`) to the input JSON and re-run Step 2. Expected: still succeeds. Confirm `video_fps` + `video_num_frames` together is rejected with the mutual-exclusion error (validates Task 1 end-to-end). If the processor rejects a kwarg name, adjust the forwarding in `omni_mot_model.py` Task 6 Step 3 (route via the processor's video-loading kwargs) and re-run.

- [ ] **Step 6: Record results**

Note in the PR description: which checkpoints were run, the generated text samples, and confirmation that image/text-only paths are unaffected.

---

## Self-review notes

- **Spec coverage:** schema fields + mutual exclusion (Task 1, spec §args), defaults (Task 2), `prepare_multimodal_reasoner_inputs` video branch (Task 3, spec §component 1), `_impl` + guards (Task 4, spec §component 2 + §validation), pass-throughs (Task 5, spec §component 3), `OmniMoTModel` video block (Task 6, spec §component 4), inference routing + batch homogeneity (Task 7, spec §component 5 + §validation), example + docs (Task 8, spec §files-touched), manual verification (Task 9, spec §verification). All spec sections mapped.
- **Naming consistency:** `pixel_values_videos` / `video_grid_thw` (model layers), `reasoner_videos` / `video_sampling_kwargs` (data_batch keys), `videos` / `video_sampling_kwargs` (`OmniMoTModel.generate_reasoner_text` params), `video_*` (sample-arg fields) — used consistently across tasks.
- **Known flag:** the exact `apply_chat_template` video-sampling kwarg surface (Task 6 Step 3) is transformers-version-dependent and confirmed in Task 9 Step 5; fallback documented inline.
- **Import paths (resolved):** `VIDEO_EXTENSIONS` is exported from `cosmos_framework.inference.common.args`; `inference.py` already imports `Path`/`Any`/`cast`/`Image`. No other new imports required.
```