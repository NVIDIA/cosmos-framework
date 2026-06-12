# Audio+Image→Video (A2V) Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `audio_image2video` inference mode that conditions video generation on a real input audio clip + an input image (first frame), using the `nano_diffusers_sound_encoder` checkpoint.

**Architecture:** Reuse the existing `ts2v` (sound-conditioned) sequence plan plus the existing image first-frame vision conditioning, which `inject_sound_into_batch` already preserves. Add a real-audio loader and a `model_mode` + `sound_path` input field; no model/network/tokenizer changes.

**Tech Stack:** Python, PyTorch, pydantic args, `soundfile` (read) + `scipy.signal` (resample) for audio I/O, `pytest` (colocated `*_test.py`), the Cosmos3 OmniMoT diffusers-format checkpoint loader. Note: `torchaudio` is NOT a project dependency and is absent from the inference container — do not use it.

**Spec:** `docs/superpowers/specs/2026-06-11-a2v-sound-encoder-inference-design.md`

**Branch:** `a2v-sound-encoder-inference`

---

## Background the implementer must know

- **No model changes.** The MOT model + AVAE already support sound-as-condition (`ts2v`). The only gap is the inference pipeline: it never loads real audio and never selects a conditioning plan.
- **`inject_sound_into_batch` preserves vision conditioning** (`cosmos_framework/inference/sound.py` ~lines 90–94, 110–111). So image-first-frame + audio-conditioned falls out of `mode="ts2v"` automatically.
- **Output decode+mux is generic** (`cosmos_framework/inference/inference.py` ~lines 1489–1544): any `"sound"` in model outputs is decoded and muxed into the `.mp4`. No change needed.
- **Sample-arg machinery:** `OmniSampleOverrides.build_sample(model_config=...)` runs `_build_vision_data` then `_build_sound_data` (`args.py` ~line 1004). `download()` methods cascade via `super().download()` through the MRO (see `VisionDataOverrides.download` at `args.py:447`).
- **Run tests** with the repo's pytest. Audio unit tests are CPU-only (synthetic tensors / tiny generated WAV). Run via the `slurm-node` skill if pytest needs the i4 container; otherwise locally.

---

## File Structure

- `cosmos_framework/inference/args.py` — checkpoint registry entry, `ModelMode.AUDIO_IMAGE2VIDEO`, `sound_path` field + validation/download.
- `cosmos_framework/inference/sound.py` — `load_conditioning_audio` helper; `condition_sound` param on `inject_sound_into_batch`.
- `cosmos_framework/inference/inference.py` — `get_sample_data` branch that loads real audio and conditions on it.
- `cosmos_framework/inference/sound_test.py` (new) — unit tests for the two `sound.py` additions.
- `cosmos_framework/inference/args_test.py` — unit test for the new mode + `sound_path` validation.
- `cosmos_framework/inference/defaults/audio_image2video/sample_args.json` (new) — per-mode defaults.
- `inputs/omni/a2v.json` (new) + `inputs/omni/assets/a2v_audio.wav` (new) — example input.
- `docs/inference.md` — Modes table row + checkpoint note.

---

## Task 1: Register the sound-encoder checkpoint

**Files:**
- Modify: `cosmos_framework/inference/args.py` (`_CHECKPOINTS`, line ~1051)
- Test: `cosmos_framework/inference/args_test.py::test_checkpoints` (existing — auto-covers the new entry)

- [ ] **Step 1: Add the registry entry**

In `_CHECKPOINTS`, after the `"Cosmos3-Nano"` entry, add:

```python
    # Diffusers HF checkpoint whose transformer is trained to condition on
    # (encode) input sound, enabling audio_image2video (A2V). Reuses the
    # Cosmos3-Nano architecture (OmniMoTModelConfig, sound_gen=True).
    "Cosmos3-Nano-SoundEncoder": CheckpointConfig(
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
        config_file=str(CONFIG_DIR / "model/Cosmos3-Nano.yaml"),
        s3_uri="",  # unused for HF-backed checkpoints
        hf=CheckpointDirHf(
            repository="nvidia/Cosmos3-Experimental",
            revision="main",
            subdirectory="nano_diffusers_sound_encoder",
        ),
    ),
```

- [ ] **Step 2: Verify the registry resolves and downloads `checkpoint.json`**

Run: `pytest cosmos_framework/inference/args_test.py::test_checkpoints -v`
Expected: PASS. (Requires `HF_TOKEN` with access to `nvidia/Cosmos3-Experimental`; run inside the dev/slurm env where the token is set. The test downloads `nano_diffusers_sound_encoder/checkpoint.json` via the `subdirectory` filter and parses it.)

- [ ] **Step 3: Commit**

```bash
git add cosmos_framework/inference/args.py
git commit -m "Register Cosmos3-Nano-SoundEncoder checkpoint"
```

---

## Task 2: Add the `audio_image2video` model mode

**Files:**
- Modify: `cosmos_framework/inference/args.py` (`ModelMode` enum line ~157; frozensets line ~181–188)
- Test: `cosmos_framework/inference/args_test.py` (new test added in Task 3, Step 5)

- [ ] **Step 1: Add the enum member**

In `class ModelMode(StrEnum)` (after `VIDEO2VIDEO = "video2video"`):

```python
    AUDIO_IMAGE2VIDEO = "audio_image2video"
```

- [ ] **Step 2: Add the mode-group frozenset and property**

After `REASONER_MODEL_MODES` (line ~188) add:

```python
# Modes that condition generation on a real input audio clip (require a model
# with ``sound_gen=True`` and a ``sound_path``).
SOUND_CONDITION_MODEL_MODES: frozenset[ModelMode] = frozenset({ModelMode.AUDIO_IMAGE2VIDEO})
```

In `class ModelMode`, alongside `is_action` / `is_reasoner`:

```python
    @property
    def is_sound_condition(self) -> bool:
        return self in SOUND_CONDITION_MODEL_MODES
```

- [ ] **Step 3: Verify import + enum value**

Run: `python -c "from cosmos_framework.inference.args import ModelMode; print(ModelMode.AUDIO_IMAGE2VIDEO.value, ModelMode.AUDIO_IMAGE2VIDEO.is_sound_condition)"`
Expected: `audio_image2video True`

- [ ] **Step 4: Commit**

```bash
git add cosmos_framework/inference/args.py
git commit -m "Add audio_image2video model mode"
```

---

## Task 3: Add the `sound_path` input field + validation/download

**Files:**
- Modify: `cosmos_framework/inference/args.py` (`SoundDataArgs` line ~514; `SoundDataOverrides` line ~518)
- Test: `cosmos_framework/inference/args_test.py`

- [ ] **Step 1: Write the failing test**

Add to `cosmos_framework/inference/args_test.py` (note the new imports `ModelMode` is already imported; add `SoundDataOverrides`):

```python
import types

from cosmos_framework.inference.args import SoundDataOverrides


def test_build_sound_data_requires_sound_path_for_a2v():
    model_config = types.SimpleNamespace(sound_gen=True)
    sample_meta = types.SimpleNamespace(model_mode=ModelMode.AUDIO_IMAGE2VIDEO)

    # Missing sound_path -> error
    overrides = SoundDataOverrides(sound_path=None)
    with pytest.raises(ValueError, match="sound_path"):
        overrides._build_sound_data(model_config=model_config, sample_meta=sample_meta)

    # sound_path set -> enable_sound forced True
    overrides = SoundDataOverrides(sound_path="clip.wav")
    overrides._build_sound_data(model_config=model_config, sample_meta=sample_meta)
    assert overrides.enable_sound is True


def test_build_sound_data_rejects_model_without_sound_gen():
    model_config = types.SimpleNamespace(sound_gen=False)
    sample_meta = types.SimpleNamespace(model_mode=ModelMode.AUDIO_IMAGE2VIDEO)
    overrides = SoundDataOverrides(sound_path="clip.wav")
    with pytest.raises(ValueError, match="sound tokenizer"):
        overrides._build_sound_data(model_config=model_config, sample_meta=sample_meta)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest cosmos_framework/inference/args_test.py::test_build_sound_data_requires_sound_path_for_a2v -v`
Expected: FAIL — `SoundDataOverrides` has no `sound_path` (TypeError/validation error).

- [ ] **Step 3: Add the field + download + validation**

In `class SoundDataArgs(ArgsBase)` (line ~514):

```python
class SoundDataArgs(ArgsBase):
    enable_sound: bool = False
    sound_path: ResolvedFilePath | None = None
```

In `class SoundDataOverrides(OverridesBase)` (line ~518) add the field and a `download()` override, and extend `_build_sound_data`:

```python
class SoundDataOverrides(OverridesBase):
    """Sound data overrides."""

    enable_sound: Training[bool | None] = None
    """Enable joint video+sound generation (t2vs mode). Requires a checkpoint with sound modules."""
    sound_path: Training[ResolvedFilePathOrUrl | None] = None
    """Path or URL to a conditioning audio clip (e.g. .wav/.mp3/.flac). Required for
    audio_image2video; the clip is encoded by the AVAE and used as a clean condition."""

    @override
    def download(self, output_dir: Path):
        super().download(output_dir)
        self.sound_path = download_file(self.sound_path, output_dir, "sound")

    def _build_sound_data(self, model_config: "OmniMoTModelConfig", sample_meta: SampleMeta):
        if sample_meta.model_mode.is_sound_condition:
            if self.sound_path is None:
                raise ValueError(
                    f"model_mode={sample_meta.model_mode.value} requires a `sound_path` "
                    "(a conditioning audio clip)"
                )
            self.enable_sound = True
        if self.enable_sound is None:
            self.enable_sound = False
        if self.enable_sound and not model_config.sound_gen:
            raise ValueError(
                "enable_sound=True requires a model with a sound tokenizer "
                "(model.config.sound_gen=True), but the loaded checkpoint has no sound tokenizer"
            )
```

(`@override`, `Path`, `download_file`, `ResolvedFilePathOrUrl`, `Training` are already imported in `args.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest cosmos_framework/inference/args_test.py::test_build_sound_data_requires_sound_path_for_a2v cosmos_framework/inference/args_test.py::test_build_sound_data_rejects_model_without_sound_gen -v`
Expected: PASS (both).

- [ ] **Step 5: Add the mode-resolution test**

Add to `args_test.py` (reuses `model_dict.config` pattern from `test_sample_args`; place it as its own test so it can build a sample for the new mode):

```python
def test_audio_image2video_conditions_image_and_sound(tmp_path: Path):
    import omegaconf
    from cosmos_framework.inference.common.config import structure_config

    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
    ).build_setup()
    model_dict = structure_config(setup_args.load_model_config_dict(), omegaconf.DictConfig)

    args = OmniSampleOverrides(
        name="a2v",
        output_dir=tmp_path / "a2v",
        model_mode=ModelMode.AUDIO_IMAGE2VIDEO,
        vision_path="robot.jpg",   # image extension -> first-frame condition
        sound_path="clip.wav",
    ).build_sample(model_config=model_dict.config)

    assert args.condition_vision_mode.value == "image"
    assert args.condition_frame_indexes_vision == [0]
    assert args.enable_sound is True
    assert args.sound_path == "clip.wav"
```

- [ ] **Step 6: Run it**

Run: `pytest cosmos_framework/inference/args_test.py::test_audio_image2video_conditions_image_and_sound -v`
Expected: PASS. (Downloads the default Nano model config; run in the dev/slurm env.)

- [ ] **Step 7: Commit**

```bash
git add cosmos_framework/inference/args.py cosmos_framework/inference/args_test.py
git commit -m "Add sound_path input + audio_image2video arg validation"
```

---

## Task 4: `load_conditioning_audio` helper

**Files:**
- Modify: `cosmos_framework/inference/sound.py`
- Test: `cosmos_framework/inference/sound_test.py` (new)

- [ ] **Step 1: Write the failing test**

Create `cosmos_framework/inference/sound_test.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import soundfile as sf
import torch

from cosmos_framework.inference.sound import load_conditioning_audio


def _write_wav(path: Path, sample_rate: int, channels: int, num_samples: int) -> None:
    data = torch.zeros(num_samples, channels).numpy() if channels > 1 else torch.zeros(num_samples).numpy()
    sf.write(str(path), data, sample_rate)


def test_load_conditioning_audio_resamples_and_pads(tmp_path: Path):
    src = tmp_path / "in.wav"
    _write_wav(src, sample_rate=44100, channels=1, num_samples=44100)  # 1.0s mono @44.1k

    out = load_conditioning_audio(src, sample_rate=48000, audio_channels=2, num_samples=96000)

    assert out.shape == (1, 2, 96000)  # [1, C, N], stereo, exactly num_samples (2.0s @48k -> pad)
    assert out.dtype == torch.float32


def test_load_conditioning_audio_trims(tmp_path: Path):
    src = tmp_path / "in.wav"
    _write_wav(src, sample_rate=48000, channels=2, num_samples=48000 * 4)  # 4s stereo @48k

    out = load_conditioning_audio(src, sample_rate=48000, audio_channels=2, num_samples=48000 * 2)

    assert out.shape == (1, 2, 48000 * 2)  # trimmed to 2s
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest cosmos_framework/inference/sound_test.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_conditioning_audio'`.

- [ ] **Step 3: Implement the helper**

Add to `cosmos_framework/inference/sound.py` (after `create_placeholder_audio`):

```python
def load_conditioning_audio(
    path: Path,
    *,
    sample_rate: int,
    audio_channels: int,
    num_samples: int,
) -> torch.Tensor:
    """Decode an audio file into a conditioning waveform aligned to the video.

    Reads ``path`` with soundfile, resamples to ``sample_rate``, conforms the
    channel count to ``audio_channels`` (mono->stereo duplicate, stereo->mono
    mean), and trims or zero-pads to exactly ``num_samples`` so the audio and
    video latent streams cover the same duration.

    Returns:
        Audio tensor of shape (1, C, N) where C == audio_channels and
        N == num_samples, dtype float32.
    """
    import soundfile as sf  # type: ignore[import-not-found]

    data, src_sr = sf.read(str(path), dtype="float32", always_2d=True)  # [N, C]
    waveform = torch.from_numpy(data).transpose(0, 1).contiguous()  # [C, N]

    # Resample to the tokenizer's rate. Uses scipy (a declared dependency);
    # torchaudio is intentionally avoided as it is not a project dependency
    # and is absent from the inference container.
    if src_sr != sample_rate:
        from math import gcd

        import scipy.signal

        g = gcd(int(src_sr), int(sample_rate))
        up, down = int(sample_rate) // g, int(src_sr) // g
        resampled = scipy.signal.resample_poly(waveform.numpy(), up, down, axis=-1)  # [C, N']
        waveform = torch.from_numpy(resampled.astype("float32")).contiguous()

    # Conform channels.
    cur_channels = waveform.shape[0]
    if cur_channels != audio_channels:
        if cur_channels == 1 and audio_channels == 2:
            waveform = waveform.repeat(2, 1)
        elif cur_channels == 2 and audio_channels == 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            raise ValueError(
                f"Cannot convert {cur_channels}-channel audio to {audio_channels} channels"
            )

    # Trim or zero-pad to num_samples.
    n = waveform.shape[-1]
    if n > num_samples:
        waveform = waveform[:, :num_samples]
    elif n < num_samples:
        waveform = torch.nn.functional.pad(waveform, (0, num_samples - n))

    return waveform.unsqueeze(0).to(dtype=torch.float32)  # [1, C, N]
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest cosmos_framework/inference/sound_test.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/inference/sound.py cosmos_framework/inference/sound_test.py
git commit -m "Add load_conditioning_audio helper"
```

---

## Task 5: `condition_sound` mode in `inject_sound_into_batch`

**Files:**
- Modify: `cosmos_framework/inference/sound.py` (`inject_sound_into_batch`, line ~62)
- Test: `cosmos_framework/inference/sound_test.py`

- [ ] **Step 1: Write the failing test**

Append to `cosmos_framework/inference/sound_test.py`:

```python
import types

from cosmos_framework.data.vfm.sequence_packing import SequencePlan


def _fake_model(sound_latent_t: int, temporal_cf: int = 4):
    sound_tok = types.SimpleNamespace(
        get_latent_num_samples=lambda n: sound_latent_t,
        audio_channels=2,
    )
    vision_tok = types.SimpleNamespace(temporal_compression_factor=temporal_cf)
    return types.SimpleNamespace(tokenizer_sound_gen=sound_tok, tokenizer_vision_gen=vision_tok)


def test_inject_sound_conditions_sound_and_preserves_image(tmp_path: Path):
    from cosmos_framework.inference.sound import inject_sound_into_batch

    model = _fake_model(sound_latent_t=50)
    # Video tensor [1,3,T,H,W] with T=48 -> 12 video latents at cf=4.
    video = torch.zeros(1, 3, 48, 16, 16)
    audio = torch.zeros(1, 2, 96000)
    batch = {
        "video": [video],
        # Pre-existing image first-frame condition (as set by build_conditioned_video_batch).
        "sequence_plan": [SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0])],
    }

    inject_sound_into_batch(batch, audio, model, condition_sound=True)

    plan = batch["sequence_plan"][0]
    assert plan.has_sound is True
    assert plan.condition_frame_indexes_sound == list(range(50))  # all sound conditioned (ts2v)
    assert plan.condition_frame_indexes_vision == [0]              # image cond preserved


def test_inject_sound_default_generates_sound(tmp_path: Path):
    from cosmos_framework.inference.sound import inject_sound_into_batch

    model = _fake_model(sound_latent_t=50)
    video = torch.zeros(1, 3, 48, 16, 16)
    audio = torch.zeros(1, 2, 96000)
    batch = {"video": [video], "sequence_plan": [SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[])]}

    inject_sound_into_batch(batch, audio, model)  # default condition_sound=False

    plan = batch["sequence_plan"][0]
    assert plan.condition_frame_indexes_sound == []  # t2vs: sound generated
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest cosmos_framework/inference/sound_test.py::test_inject_sound_conditions_sound_and_preserves_image -v`
Expected: FAIL — `inject_sound_into_batch` got an unexpected keyword argument `condition_sound`.

- [ ] **Step 3: Add the `condition_sound` parameter**

In `cosmos_framework/inference/sound.py`, change the signature and the mode selection:

```python
def inject_sound_into_batch(
    data_batch: dict[str, Any],
    audio_tensor: torch.Tensor | None,
    model: Any,
    *,
    condition_sound: bool = False,
) -> dict[str, Any]:
```

Update the docstring's Args to add:

```
        condition_sound: When True, the provided audio is used as a clean
            condition (mode "ts2v") and the video is generated from it. When
            False (default), sound is generated jointly (mode "t2vs").
```

In the `if has_sound:` block, replace the hardcoded `mode="t2vs"`:

```python
        sequence_plan = build_sequence_plan_for_sound(
            mode="ts2v" if condition_sound else "t2vs",
            video_latent_length=video_latent_t,
            sound_latent_length=sound_latent_t,
        )
```

(The existing `existing_vision_cond` preservation below it already re-applies the image's `condition_frame_indexes_vision`.)

- [ ] **Step 4: Run to verify both pass**

Run: `pytest cosmos_framework/inference/sound_test.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/inference/sound.py cosmos_framework/inference/sound_test.py
git commit -m "Support sound conditioning (ts2v) in inject_sound_into_batch"
```

---

## Task 6: Wire real audio into `get_sample_data`

**Files:**
- Modify: `cosmos_framework/inference/inference.py` (`get_sample_data`, line ~610)

- [ ] **Step 1: Replace the sound-injection block**

In `get_sample_data`, replace the existing `if sample_args.enable_sound:` block (lines ~610–625) with:

```python
        if sample_args.enable_sound:
            from cosmos_framework.inference.sound import (
                create_placeholder_audio,
                get_audio_tokenizer_info,
                inject_sound_into_batch,
                load_conditioning_audio,
            )

            audio_info = get_audio_tokenizer_info(model)
            if not audio_info.has_sound:
                raise ValueError("enable_sound=True but model has no sound tokenizer")

            condition_sound = sample_args.sound_path is not None
            if condition_sound:
                num_samples = int(sample_args.num_frames / sample_args.fps * audio_info.sample_rate)
                audio = load_conditioning_audio(
                    Path(sample_args.sound_path),
                    sample_rate=audio_info.sample_rate,
                    audio_channels=getattr(audio_info.tokenizer, "audio_channels", 2),
                    num_samples=num_samples,
                )
            else:
                audio = create_placeholder_audio(
                    num_frames=sample_args.num_frames,
                    conditioning_fps=sample_args.fps,
                    audio_info=audio_info,
                )
            inject_sound_into_batch(out, audio, model, condition_sound=condition_sound)
```

(`Path` is already imported in `inference.py`.)

- [ ] **Step 2: Sanity-check the module imports**

Run: `python -c "import cosmos_framework.inference.inference"`
Expected: no error (module imports cleanly).

- [ ] **Step 3: Commit**

```bash
git add cosmos_framework/inference/inference.py
git commit -m "Load and condition on real input audio in get_sample_data"
```

---

## Task 7: Defaults + example input + conditioning audio asset

**Files:**
- Create: `cosmos_framework/inference/defaults/audio_image2video/sample_args.json`
- Create: `inputs/omni/assets/a2v_audio.wav`
- Create: `inputs/omni/a2v.json`

- [ ] **Step 1: Create the per-mode defaults**

Create `cosmos_framework/inference/defaults/audio_image2video/sample_args.json` (image2video defaults + `enable_sound: true`):

```json
{
    "num_steps": 35,
    "guidance": 6.0,
    "shift": 10.0,
    "sigma_max": 80.0,
    "normalize_cfg": false,
    "autoregressive": false,
    "negative_prompt": null,
    "negative_prompt_file": "neg_prompts.json",
    "duration_template": "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS.",
    "resolution_template": "This video is of {height}x{width} resolution.",
    "negative_metadata_mode": "none",
    "inverse_duration_template": "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS.",
    "inverse_resolution_template": "This video is not of {height}x{width} resolution.",
    "negative_prompt_keep_metadata": true,
    "aspect_ratio": "16,9",
    "fps": 24,
    "num_frames": 189,
    "video_save_quality": 10,
    "image_save_quality": 95,
    "enable_sound": true
}
```

- [ ] **Step 2: Extract a conditioning audio clip from a published example**

Run inside the i4 container (via the `slurm-node` skill), from the repo root:

```bash
mkdir -p inputs/omni/assets
curl -sS -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/nvidia/Cosmos3-Nano/resolve/main/assets/example_t2vs_output.mp4" \
  -o /tmp/example_t2vs_output.mp4
python - <<'PY'
import av, numpy as np, soundfile as sf
container = av.open("/tmp/example_t2vs_output.mp4")
astream = container.streams.audio[0]
sr = astream.codec_context.sample_rate
frames = [f.to_ndarray() for f in container.decode(astream)]  # each [C, n] fltp or [n*C] packed
# Normalize to [N, C] float32.
import numpy as np
chunks = []
for arr in frames:
    if arr.ndim == 2:           # planar [C, n]
        chunks.append(arr.T)
    else:                        # packed
        chunks.append(arr.reshape(-1, astream.channels))
audio = np.concatenate(chunks, axis=0).astype("float32")
audio = audio[: sr * 4]          # keep first 4 seconds to bound file size
sf.write("inputs/omni/assets/a2v_audio.wav", audio, sr)
print("wrote", audio.shape, "@", sr)
PY
```

Expected: prints the shape and writes `inputs/omni/assets/a2v_audio.wav` (~1–1.5 MB).

- [ ] **Step 3: Create the example input file**

Create `inputs/omni/a2v.json` (image from the existing i2v example; audio is the extracted clip, referenced relative to the input file):

```json
{
  "model_mode": "audio_image2video",
  "name": "a2v",
  "prompt": "{\"temporal_caption\": \"A silver robotic arm in a clean lab pours water from a glass jar into a white ceramic cup; soft mechanical whirring and gentle water trickling are audible.\", \"audio_description\": \"Gentle splashing and trickling of water with a faint mechanical whir from servo motors; no speech or music.\", \"resolution\": {\"H\": 720, \"W\": 1280}, \"aspect_ratio\": \"16,9\", \"fps\": 24}",
  "vision_path": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_153.jpg",
  "sound_path": "assets/a2v_audio.wav"
}
```

- [ ] **Step 4: Validate the example parses into sample args**

Run inside the dev/slurm env:

```bash
python -c "
import json, pathlib
d = json.loads(pathlib.Path('inputs/omni/a2v.json').read_text())
assert d['model_mode'] == 'audio_image2video'
assert d['sound_path'].endswith('.wav')
print('ok')
"
```
Expected: `ok`. (Full arg-building is exercised by the slurm run in Task 9.)

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/inference/defaults/audio_image2video/sample_args.json inputs/omni/a2v.json inputs/omni/assets/a2v_audio.wav
git commit -m "Add audio_image2video defaults and a2v example input"
```

---

## Task 8: Documentation

**Files:**
- Modify: `docs/inference.md`

- [ ] **Step 1: Add the Modes table row**

In the Modes table (after the `video2video` row), add:

```markdown
| `audio_image2video` | text prompt + image + audio                | `vision.mp4` (with conditioning audio muxed in)                                            | `prompt`, `vision_path`, `sound_path`       | [`inputs/omni/a2v.json`](../inputs/omni/a2v.json)                                                                                                                                                                                                                                                                                                                                                                       |
```

- [ ] **Step 2: Note the checkpoint under the sound sentence**

Replace the existing sentence (line ~148):

```markdown
Set `enable_sound: true` on a `text2video` sample (see [`inputs/omni/t2vs.json`](../inputs/omni/t2vs.json)) to also generate audio. To run every example in one batch, use `-i "inputs/omni/*.json"`.
```

with:

```markdown
Set `enable_sound: true` on a `text2video` sample (see [`inputs/omni/t2vs.json`](../inputs/omni/t2vs.json)) to also generate audio. To instead **condition** generation on a real audio clip (audio+image → video), use `model_mode: audio_image2video` with a `sound_path` and the sound-encoder checkpoint `--checkpoint-path Cosmos3-Nano-SoundEncoder` (see [`inputs/omni/a2v.json`](../inputs/omni/a2v.json)). To run every example in one batch, use `-i "inputs/omni/*.json"`.
```

- [ ] **Step 3: Add the checkpoint to the Models table**

In the Models table, add a row:

```markdown
| Cosmos3-Nano-SoundEncoder | `--checkpoint-path=Cosmos3-Nano-SoundEncoder` | `audio_image2video` (audio+image → video), plus all Nano modes |
```

- [ ] **Step 4: Commit**

```bash
git add docs/inference.md
git commit -m "Document audio_image2video mode and sound-encoder checkpoint"
```

---

## Task 9: End-to-end A2V run on slurm (verification gate)

**Files:** none (verification only)

- [ ] **Step 1: Run A2V inference** via the `slurm-node` skill, from the repo root inside the i4 container:

```bash
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/a2v.json" \
    -o outputs/a2v \
    --checkpoint-path Cosmos3-Nano-SoundEncoder \
    --seed=0
```

Expected: completes without shape/dtype errors during sound encode/condition; writes `outputs/a2v/a2v/vision.mp4` and `sample_args.json`.

- [ ] **Step 2: Verify the output**

```bash
python - <<'PY'
import av
c = av.open("outputs/a2v/a2v/vision.mp4")
assert len(c.streams.video) == 1, "no video stream"
assert len(c.streams.audio) >= 1, "no audio track muxed"
print("video frames:", c.streams.video[0].frames, "| audio streams:", len(c.streams.audio))
PY
```
Expected: prints a positive video frame count and ≥1 audio stream.

- [ ] **Step 3: Inspect `sample_args.json`** to confirm conditioning was applied:

```bash
python -c "
import json
a = json.load(open('outputs/a2v/a2v/sample_args.json'))
assert a['model_mode'] == 'audio_image2video'
assert a['enable_sound'] is True
assert a['condition_frame_indexes_vision'] == [0]
print('conditioning ok')
"
```
Expected: `conditioning ok`.

- [ ] **Step 4: If sound tokens are not frozen during sampling** (risk #1 in the spec — e.g. output audio does not match the input clip), STOP and use `superpowers:systematic-debugging` to trace whether the diffusion loop respects `condition_frame_indexes_sound`. This is the one place a model-side fix might be needed; do not paper over it.

---

## Self-Review

**Spec coverage:**
- Checkpoint registration → Task 1. ✅
- `audio_image2video` mode → Task 2. ✅
- `sound_path` field + validation/download → Task 3. ✅
- `load_conditioning_audio` → Task 4. ✅
- `condition_sound` wiring (`ts2v` + preserved image cond) → Tasks 5–6. ✅
- Defaults + example + audio asset → Task 7. ✅
- Output decode+mux (no change) → confirmed in Task 9 verification. ✅
- Tests (colocated) → Tasks 3, 4, 5. ✅ (`sound_data_utils` `ts2v` is exercised indirectly by Task 5's plan assertions; the spec's optional `sound_data_utils_test.py` is dropped as redundant — Task 5 already asserts the resulting `condition_frame_indexes_sound`.)
- Docs → Task 8. ✅
- E2E slurm verification → Task 9. ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✅

**Type consistency:** `load_conditioning_audio(path, *, sample_rate, audio_channels, num_samples) -> Tensor[1,C,N]` is defined in Task 4 and called identically in Task 6. `inject_sound_into_batch(..., *, condition_sound=False)` defined in Task 5, called with `condition_sound=condition_sound` in Task 6. `ModelMode.AUDIO_IMAGE2VIDEO` / `is_sound_condition` / `SOUND_CONDITION_MODEL_MODES` consistent across Tasks 2–3. ✅

**Note on `sound_data_utils_test.py`:** The spec listed it as "if absent." Dropped to avoid duplicate coverage; Task 5 asserts the `ts2v` plan's effect end-to-end through `inject_sound_into_batch`. If a direct unit test is preferred, add one asserting `build_sequence_plan_for_sound("ts2v", v, s).condition_frame_indexes_sound == list(range(s))`.
