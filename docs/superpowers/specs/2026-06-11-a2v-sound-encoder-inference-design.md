# Audio+Image→Video (A2V) Inference with the Sound-Encoder Checkpoint

**Date:** 2026-06-11
**Status:** Approved design — ready for implementation plan
**Branch:** `a2v-sound-encoder-inference`

## Summary

Add a new inference mode, `audio_image2video`, that conditions video generation on a
**real input audio clip** plus an **input image (first frame)**, using the
`nano_diffusers_sound_encoder` checkpoint
(`nvidia/Cosmos3-Experimental`, subfolder `nano_diffusers_sound_encoder`).

Today the only audio path in inference is *generation*: `enable_sound: true` injects a
zero-filled placeholder waveform and the model **generates** sound (internally
`mode="t2vs"`). There is no way to feed a real audio clip as a **condition**. This spec
closes that gap on the inference side only — the model and data layers already support
sound-as-condition.

## Background / Current State

- **Checkpoint format.** The default `Cosmos3-Nano` checkpoint is diffusers-format
  (`model_index.json`, `transformer/`, `vae/`, `sound_tokenizer/`, `vision_encoder/`)
  and already ships a `sound_tokenizer` (AVAE). The framework's loader already handles
  this format. The new `nano_diffusers_sound_encoder` checkpoint has the same layout
  with different `transformer` + `sound_tokenizer` weights; its `config.json` maps to the
  same `OmniMoTModelConfig` (`sound_gen=true`, `sound_dim=64`, AVAE
  `sample_rate=48000`, `audio_channels=2`).
- **Sound generation today.** `cosmos_framework/inference/inference.py:get_sample_data`
  (≈ line 610) calls `create_placeholder_audio` → `inject_sound_into_batch`, which
  hardcodes `mode="t2vs"` in `cosmos_framework/inference/sound.py:105`. All sound tokens
  are generated; the placeholder only establishes the target length.
- **Sound conditioning already exists at the data/model layer.**
  `cosmos_framework/data/vfm/sound_data_utils.py` defines `ts2v` (Text+Sound→Video,
  sound conditioned) and `ti2sv`. The MOT network handles clean vs. noisy sound tokens
  via the condition mask (`_encode_sound`/`_decode_sound`). These plans are **not
  reachable** from any `model_mode`.
- **Vision-condition preservation.** `inject_sound_into_batch` captures and re-applies
  the existing `condition_frame_indexes_vision` (sound.py lines ~90–94, 110–111). So the
  image's first-frame condition survives whichever sound mode is chosen.
- **Output path is generic.** `inference.py` (≈ lines 1489–1544) decodes any `"sound"` in
  the model outputs (`model.decode_sound`) and muxes it into the `.mp4`
  (`mux_audio_into_video`). No change is needed for conditioned audio.

### Current `ModelMode` values

`text2image`, `text2video`, `image2image`, `image2video`, `video2video`,
`forward_dynamics`, `inverse_dynamics`, `policy`, `reasoner`. None takes audio as input.

## Goals

1. Register the `nano_diffusers_sound_encoder` checkpoint as a named checkpoint.
2. Add `audio_image2video`: image (first frame) + real audio clip → video, with the audio
   used as a **clean condition** and muxed into the output video.
3. Full deliverable: input loading, defaults, example input, colocated tests, docs.

## Non-Goals

- Audio-only conditioning (`ts2v` with no image) as a separate user-facing mode (YAGNI).
- Any change to the model architecture, sequence packing, AVAE tokenizer, training, or
  the output/save path.
- A new `tis2v` sequence plan: the audio+image combination falls out of `ts2v` (conditions
  sound) plus the preserved image vision-condition.

## Design

### Key leverage point

"Audio+image→video" needs no new sequence-plan combination. `inject_sound_into_batch`
already preserves the image's `condition_frame_indexes_vision=[0]`. Selecting the existing
`ts2v` plan (all sound latents conditioned) and letting that preservation re-apply `[0]`
yields exactly: image first-frame conditioned + audio conditioned + remaining video
generated.

### Changes by component

**1. Checkpoint registry** — `cosmos_framework/inference/args.py` (`_CHECKPOINTS`, line 1051)

Add:

```python
"Cosmos3-Nano-SoundEncoder": CheckpointConfig(
    model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
    config_file=str(CONFIG_DIR / "model/Cosmos3-Nano.yaml"),
    s3_uri=...,  # match the Nano entry's pattern; unused for HF-backed load
    hf=CheckpointDirHf(
        repository="nvidia/Cosmos3-Experimental",
        revision="main",
        subdirectory="nano_diffusers_sound_encoder",
    ),
)
```

Reuses `Cosmos3-Nano.yaml` (same `OmniMoTModelConfig`). Weight compatibility is verified
on load and end-to-end by the slurm run.

**2. New model mode** — `cosmos_framework/inference/args.py` (`ModelMode`, line 157)

- Add `AUDIO_IMAGE2VIDEO = "audio_image2video"`.
- Add a `_SOUND_CONDITION_MODES: frozenset[ModelMode] = {ModelMode.AUDIO_IMAGE2VIDEO}` and
  an `is_sound_condition` property mirroring `is_action`/`is_reasoner`.
- `condition_vision_mode` already resolves to `image` from an image `vision_path`
  (lines 355–368), giving `condition_frame_indexes_vision=[0]` via existing defaults.

**3. Audio input field** — `cosmos_framework/inference/args.py`

- `SoundDataArgs` (line 514): add `sound_path: ResolvedFilePath | None = None`.
- `SoundDataOverrides` (line 518): add `sound_path: ResolvedFilePathOrUrl | None = None`
  with a docstring; add a `download()` override that calls
  `self.sound_path = download_file(self.sound_path, output_dir, "sound")`.
- `_build_sound_data` (line 524): for `AUDIO_IMAGE2VIDEO`, require `sound_path` set and
  force `enable_sound = True`; keep the existing `sound_gen` validation.

**4. Audio decode helper** — `cosmos_framework/inference/sound.py`

```python
def load_conditioning_audio(
    path: Path,
    *,
    sample_rate: int,
    audio_channels: int,
    num_samples: int,
) -> torch.Tensor:
    """Decode an audio file to a [1, C, N] waveform aligned to the video duration.

    Reads via soundfile, resamples to ``sample_rate``, conforms channel count to
    ``audio_channels`` (mono->stereo duplicate / stereo->mono mean), and trims or
    zero-pads to ``num_samples`` so the audio and video latent streams align temporally.
    """
```

`num_samples` = `int(num_frames / fps * sample_rate)` (matches `create_placeholder_audio`).

**5. Wire conditioning**

- `cosmos_framework/inference/sound.py`: add `condition_sound: bool = False` to
  `inject_sound_into_batch`. When `True`, build the plan with `mode="ts2v"` (instead of
  `"t2vs"`); the existing vision-cond preservation keeps the image's `[0]`.
- `cosmos_framework/inference/inference.py` `get_sample_data` (≈ line 610): when
  `sample_args.sound_path` is set, call `load_conditioning_audio(...)` and
  `inject_sound_into_batch(out, audio, model, condition_sound=True)`. Otherwise keep the
  existing placeholder-generation branch unchanged.

**6. Defaults + example**

- `cosmos_framework/inference/defaults/audio_image2video/sample_args.json`: copy of
  `image2video/sample_args.json` with `"enable_sound": true`.
- `inputs/omni/a2v.json`: `model_mode="audio_image2video"`, image `vision_path`,
  `sound_path`, and a prompt.

**7. Output** — no code change. `inference.py:1489` decodes and muxes the (clean,
conditioned) sound into `vision.mp4`.

### Data flow

```
a2v.json (vision_path=image, sound_path=audio, model_mode=audio_image2video)
  -> download() resolves both paths
  -> get_sample_data:
       condition_vision_mode=image -> load_conditioning_image -> build_conditioned_video_batch
           (condition_frame_indexes_vision=[0], sequence_plan with image cond)
       sound_path set -> load_conditioning_audio -> [1,C,N] real waveform
       inject_sound_into_batch(condition_sound=True):
           mode="ts2v" -> condition_frame_indexes_sound = all
           preserve condition_frame_indexes_vision=[0]
  -> model encodes audio (AVAE) as clean sound tokens; image as clean first frame
  -> diffusion generates remaining video frames conditioned on both
  -> outputs["sound"] decoded + muxed into vision.mp4
```

## Testing

Colocated unit tests (matching the `colocated-tests-ci` convention):

- ➕ `cosmos_framework/inference/sound_test.py`
  - `load_conditioning_audio`: resample rate change, mono↔stereo conformance,
    trim/pad to `num_samples`, returns `[1, C, num_samples]`.
  - `inject_sound_into_batch(condition_sound=True)`: produces a plan with all sound
    latents conditioned **and** preserves a pre-set `condition_frame_indexes_vision=[0]`;
    `condition_sound=False` keeps the existing `t2vs` behavior.
- ✏️ `cosmos_framework/inference/args_test.py`
  - `audio_image2video` resolves `condition_vision_mode=image` and
    `condition_frame_indexes_vision=[0]`.
  - `_build_sound_data` requires `sound_path` and sets `enable_sound=True`; validation
    fails on a model with `sound_gen=False`.
- ➕ `cosmos_framework/data/vfm/sound_data_utils_test.py` (if absent): `ts2v` plan sets
  `condition_frame_indexes_sound = range(sound_latent_length)` and empty vision cond.

CPU-only tests use small synthetic tensors / a tiny generated `.wav`; no GPU or checkpoint
download required.

## End-to-end verification (slurm)

Run at repo root inside the i4 container (per the `slurm-node` skill):

```shell
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/a2v.json" \
    -o outputs/a2v \
    --checkpoint-path Cosmos3-Nano-SoundEncoder \
    --seed=0
```

Success = `outputs/a2v/<sample>/vision.mp4` exists, plays, and contains an audio track
matching the input clip; no shape/dtype errors during sound encode/condition.

## Example inputs (sourced by implementer)

- **Image:** the robot image already referenced by `inputs/omni/i2v.json`.
- **Audio:** extract the audio track from a Cosmos3-Nano example sound output
  (e.g. `assets/example_t2vs_output.mp4` in the `nvidia/Cosmos3-Nano` HF repo) into a
  short `.wav`, referenced by URL or local path.

## Risks

1. **Conditioned sound tokens must stay frozen during sampling.** The model uses the same
   forward path as `ts2v` training, so this should hold; the slurm run is the gate. If the
   sampler does not freeze clean sound tokens, a small inference-loop fix may be needed
   (out of scope until observed).
2. **Checkpoint/config compatibility.** `nano_diffusers_sound_encoder/config.json` matches
   `Cosmos3-Nano.yaml`'s `OmniMoTModelConfig`; confirmed by inspection, verified on load.
3. **Audio/video temporal alignment.** Handled by trimming/padding audio to the video
   duration in `load_conditioning_audio`.

## Files touched (summary)

Modify: `inference/args.py`, `inference/sound.py`, `inference/inference.py`,
`inference/args_test.py`, `docs/inference.md`.
Create: `inference/sound_test.py`, `inference/defaults/audio_image2video/sample_args.json`,
`inputs/omni/a2v.json`, and `data/vfm/sound_data_utils_test.py` (if absent).
