# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Minimal training demo — drive Cosmos's OmniMoTModel from a plain PyTorch loop.

⚠  THIS IS A WIRING DEMO. It shows the smallest possible call sequence to drive
   `model.training_step` from your own loop — it is NOT a fine-tuning recipe.
   Production SFT uses FSDP across ≥ 8 GPUs (AdamW), real datasets (not the
   random tensors used here), and a curriculum / callbacks / EMA. The TOML
   recipes in `examples/toml/*.toml` are the real entry points.

⚠  THE MAIN TRANSFORMER IS RANDOM-INITIALIZED — the demo never loads the
   ~30 GB Cosmos3-Nano DCP shards. Loss values are therefore meaningless;
   the point is to show the call sequence and tensor shapes. For real weight
   loading see `cosmos_framework.inference.model.Cosmos3OmniModel.from_pretrained_dcp`
   and the production trainer in `cosmos_framework.scripts.train`.

================================================================================
SCOPE
================================================================================
This is NOT "extracting the model into another framework". The cosmos_framework package
must be installed (`pip install -e .` from the repo root). OmniMoTModel has deep
imports across cosmos_framework (sequence packing, MoE network, VAE, …) — physically
excising it isn't realistic.

What this demo SHOWS is the integration contract:
    - what to import,
    - what the input batch dict must contain,
    - which model methods to call,
so that you can plug OmniMoTModel into your own training framework as a
black-box `nn.Module` whose `training_step` returns a scalar loss.

What we USE from cosmos_framework:
    cosmos_framework.inference.model.Cosmos3OmniModel          → model class (random-init in this demo;
                                                       use `.from_pretrained_dcp(...)` for real weights)
    cosmos_framework.inference.common.init.init_script         → 1-line torch.distributed init
    cosmos_framework.model.generator.reasoner.qwen3_vl.utils.tokenize_caption
                                                     → text tokenizer (modelling pkg)
    model.training_step(batch, iteration)            → THE training step (flow-matching loss)
    model.config.{action_gen,sound_gen,vision_gen,…} → modality flags

What we DO NOT use:
    cosmos_framework.scripts.train, cosmos_framework.trainer.*           → CLI + Trainer class
    cosmos_framework.data.generator.joint_dataloader.*               → iterative joint dataloader
    cosmos_framework.data.generator.augmentor_provider.*             → text/video augmentor pipeline
    cosmos_framework.inference.inference.OmniInference          → inference pipeline

================================================================================
WHY init_script() IS NEEDED
================================================================================
OmniMoTModel uses torch.distributed primitives even on a single GPU
(ParallelDims, DTensor helpers, FSDP composables). `init_script()` runs
`torch.distributed.init_process_group("nccl")` in 1-rank mode and registers DCP
config wrappers. Drop it and the loader crashes with cryptic "default process
group not initialized" errors.

================================================================================
DATA BATCH CONTRACT (single-modality vision branch)
================================================================================
The dict passed to `model.training_step(batch, iteration)` must contain:

    Key                            Type                       Shape / Notes
    ────────────────────────────────────────────────────────────────────────
    model.input_video_key          list[Tensor]  (len=B)      [1, C=3, T, H, W] in [-1, 1]
        (default: "video")                                    For T>1, video; for T=1, image.
    model.input_image_key          list[Tensor]  (len=B)      [1, C=3, 1, H, W] in [-1, 1]
        (default: "images")                                   Alternative image-only entry point.
    model.input_caption_key        list[str]     (len=B)      raw text (NOT re-tokenized by model)
        (default: "ai_caption")
    "text_token_ids"               list[Tensor]  (len=B)      [1, N_tok] long tensor — pre-tokenized
    "image_size"                   list[Tensor]  (len=B)      [1, 4] float — (H, W, H, W)
    "fps"                          Tensor                     [B]  float
    "conditioning_fps"             Tensor                     [B]  float
    "num_frames"                   Tensor                     [B]  int
    "is_preprocessed"              bool                       True ⇒ video already normalized

For ACTION training (forward dynamics / policy) the batch also needs `action`,
`domain_id`, `raw_action_dim`, `mode`, and a hand-built `sequence_plan` — see
`make_action_fdm_batch` below for a worked example, or
`cosmos_framework/inference/action.py: build_action_batch` for the canonical impl.

GOTCHA — video shape differs between training and inference batches:
    Training (this file, is_preprocessed=True) expects a FLAT list of tensors:
        batch[model.input_video_key] = [video]              # [1, C, T, H, W]
    Inference (`cosmos_framework.inference.action.build_action_batch`) uses NESTED:
        batch[model.input_video_key] = [[video]]            # one extra []
    Copying an inference batch into a training loop produces a confusing
    `_normalize_video_databatch_inplace` error. Use the flat convention here.

================================================================================
MEMORY (READ THIS BEFORE RUNNING)
================================================================================
Full-fine-tuning the 8B Cosmos3-Nano on a single 80 GB GPU does NOT fit with
AdamW (param + grad + Adam moments ≈ 96 GB). For a single-GPU demo we use SGD
(no optimizer state) and small inputs; full SFT in production uses FSDP across
≥ 8 GPUs and/or LoRA — see `cosmos_framework.scripts.train` and `examples/toml/*.toml`.

To make full-fine-tuning fit on real hardware, you would either:
    - shard with FSDP (`cosmos_framework.utils.generator.parallelism.ParallelDims` + FSDP wrap),
    - inject LoRA (`model.add_lora(...)`), or
    - swap the optimizer for one with lower state (Adafactor, 8-bit AdamW).

================================================================================
RUN
================================================================================
    PYTHONPATH=. python examples/integration/trainer_level_training.py
    PYTHONPATH=. python examples/integration/trainer_level_training.py --config-dir /path/to/dir/with/config.json
"""

from cosmos_framework.inference.common.init import init_script

init_script(training=True)  # ← see docstring above

import argparse
import json
from pathlib import Path

import attrs
import torch

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.args import DEFAULT_CHECKPOINT
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption


def _load_omni_model(*, config_dir_arg: str | None):
    """Build OmniMoTModel with RANDOM main-transformer weights — wiring demo only.

    This helper exists so the demo can run without downloading the ~30 GB transformer
    DCP. Only ``config.json`` is fetched (single ~5 KB file) and the main net is
    instantiated via ``hydra.utils.instantiate`` with random parameters. Auxiliary
    sub-models (Qwen3-VL tokenizer, Wan2.2 VAE, AVAE) still load from the HF cache
    during ``Cosmos3OmniModel.__init__`` — they are not stubbed out.

    For REAL weight loading, see
    :func:`cosmos_framework.inference.model.Cosmos3OmniModel.from_pretrained_dcp`
    and the production trainer in :mod:`cosmos_framework.scripts.train`.
    """
    if config_dir_arg is None:
        from huggingface_hub import hf_hub_download
        config_dir = Path(hf_hub_download(
            repo_id=DEFAULT_CHECKPOINT.hf.repository,
            filename="config.json",
            revision=DEFAULT_CHECKPOINT.hf.revision,
        )).parent
    else:
        config_dir = Path(config_dir_arg)
    # Shipped DCPs nest config.json one level deeper under model/.
    if not (config_dir / "config.json").exists() and (config_dir / "model" / "config.json").exists():
        config_dir = config_dir / "model"
    print(f"Loading config from: {config_dir / 'config.json'}")

    # Shipped configs carry stale `cosmos3._src.*` dotted module strings in `_type` / `_target_`
    # fields. cosmos_framework's CONFIG_REPLACEMENTS_INVERSE only rewrites the slash-form
    # paths, so we rewrite the dotted form here before constructing the config.
    config_text = (config_dir / "config.json").read_text()
    for _old, _new in [
        ("cosmos3._src.vfm.configs.base.", "cosmos_framework.configs.base."),
        ("cosmos3._src.vfm.models.", "cosmos_framework.model.generator."),
        ("cosmos3._src.vfm.tokenizers.", "cosmos_framework.model.generator.tokenizers."),
        ("cosmos3._src.imaginaire.", "cosmos_framework."),
    ]:
        config_text = config_text.replace(_old, _new)
    config = Cosmos3OmniConfig(model=json.loads(config_text)["model"])
    config.parallelism = attrs.asdict(ParallelismConfig())
    config.compile = attrs.asdict(CompileConfig(enabled=False))
    return Cosmos3OmniModel(config).model


# ────────────────────────────────────────────────────────────────────────────
# Per-modality batch builders. Each returns a B=1 dict in the shape that
# model.training_step expects. Plug your own dataset in by producing the
# same keys per sample and collating into list-valued entries.
# ────────────────────────────────────────────────────────────────────────────

def _tokenize(model, caption: str, device) -> torch.Tensor:
    """Tokenize a caption using the model's own VLM tokenizer."""
    ids = tokenize_caption(
        caption,
        model.vlm_tokenizer,
        is_video=False,
        use_system_prompt=model.vlm_config.use_system_prompt,
    )
    # Shape [1, N_tok]. The collate format in cosmos_framework.data.generator.joint_dataloader
    # keeps text_token_ids as a list of [1, N] tensors (one per sample) because
    # token counts vary across the batch.
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def make_text_to_image_batch(model, *, caption: str, h: int = 128, w: int = 128, device="cuda") -> dict:
    """Text-to-image: vision branch with T=1."""
    image = (torch.randn(1, 3, 1, h, w, device=device) * 0.3).clamp(-1, 1)  # must be in [-1, 1]
    return {
        model.input_image_key:   [image],                                                       # T=1 → image branch
        model.input_caption_key: [caption],
        "text_token_ids":        [_tokenize(model, caption, device)],
        "image_size":            [torch.tensor([[h, w, h, w]], dtype=torch.float32, device=device)],
        "fps":              torch.tensor([16.0], device=device),
        "conditioning_fps": torch.tensor([16.0], device=device),
        "num_frames":       torch.tensor([1], device=device),
        "is_preprocessed":  True,
    }


def make_text_to_video_batch(model, *, caption: str, num_frames: int = 17,
                             h: int = 128, w: int = 128, device="cuda") -> dict:
    """Text-to-video: vision branch with T>1. Same model, same loss — only T differs."""
    video = (torch.randn(1, 3, num_frames, h, w, device=device) * 0.3).clamp(-1, 1)
    return {
        model.input_video_key:   [video],
        model.input_caption_key: [caption],
        "text_token_ids":        [_tokenize(model, caption, device)],
        "image_size":            [torch.tensor([[h, w, h, w]], dtype=torch.float32, device=device)],
        "fps":              torch.tensor([16.0], device=device),
        "conditioning_fps": torch.tensor([16.0], device=device),
        "num_frames":       torch.tensor([num_frames], device=device),
        "is_preprocessed":  True,
    }


def make_sound_video_batch(model, *, caption: str, num_video_frames: int = 5,
                           audio_hop_count: int = 8, h: int = 128, w: int = 128,
                           device="cuda") -> dict:
    """Joint text→video+sound batch (t2vs mode).

    Requires `model.config.sound_gen=True`. The model's AVAE expects stereo
    audio at 48 kHz with hop_size=1920 (Cosmos3-Nano defaults), so we round
    `num_audio_samples = audio_hop_count * 1920`. Audio and video duration
    don't have to match exactly; cosmos_framework handles temporal alignment via RoPE
    fps modulation in `_get_sound_fps_for_rope`.
    """
    # Stereo (AVAE expects 2 channels). 8 hops × 1920 = 15360 samples = 0.32 s @ 48 kHz.
    audio_channels = 2
    num_audio_samples = audio_hop_count * 1920
    waveform = (torch.randn(audio_channels, num_audio_samples, device=device) * 0.1).clamp(-1, 1)

    video = (torch.randn(1, 3, num_video_frames, h, w, device=device) * 0.3).clamp(-1, 1)

    # Sequence plan has both vision and sound; default condition indexes ([]) mean
    # all frames / all sound latent steps are noised and supervised.
    sequence_plan = SequencePlan(
        has_text=True,
        has_vision=True,
        has_sound=True,
    )

    return {
        model.input_video_key:   [video],
        "sound":                 [waveform],
        model.input_caption_key: [caption],
        "text_token_ids":        [_tokenize(model, caption, device)],
        "image_size":            [torch.tensor([[h, w, h, w]], dtype=torch.float32, device=device)],
        "fps":              torch.tensor([16.0], device=device),
        "conditioning_fps": torch.tensor([16.0], device=device),
        "num_frames":       torch.tensor([num_video_frames], device=device),
        "sequence_plan":    [sequence_plan],
        "is_preprocessed":  True,
    }


def make_action_fdm_batch(model, *, caption: str, num_video_frames: int = 5,
                          action_chunk: int = 4, raw_action_dim: int = 7,
                          h: int = 128, w: int = 128,
                          domain_name: str = "bridge_orig_lerobot", device="cuda") -> dict:
    """Action forward-dynamics: predict future video given 1st frame + action sequence.

    Requires `model.config.action_gen=True`. The batch contract is a superset of
    the vision batch: the same `video` / text fields plus an `action` tensor, a
    `domain_id` (cross-embodiment routing), `raw_action_dim` (un-padded dim;
    cosmos_framework pads to `max_action_dim`), `mode`, and a hand-built `sequence_plan`.
    See `cosmos_framework/inference/action.py: build_action_batch` for the canonical impl.

    `domain_name` selects the cross-embodiment routing; see
    `cosmos_framework/data/generator/action/domain_utils.py` for the full list of supported
    embodiments.
    """
    # First frame is the conditioning anchor; remaining frames are predicted.
    video = (torch.randn(1, 3, num_video_frames, h, w, device=device) * 0.3).clamp(-1, 1)  # [1, C, T, H, W]

    # Pad raw action (e.g. 7-DoF: xyz + rpy + gripper) to max_action_dim.
    action = torch.zeros(action_chunk, model.config.max_action_dim, device=device)
    action[:, :raw_action_dim] = torch.randn(action_chunk, raw_action_dim, device=device) * 0.1

    # Hand-built sequence plan tells the packer which frames are conditioning.
    sequence_plan = build_sequence_plan_from_mode(
        mode="forward_dynamics",
        video_length=num_video_frames,
        action_length=action_chunk,
        has_text=True,
    )

    # Note: the inference-side `build_action_batch` uses `[[video]]` (nested) but
    # the training-side _normalize_video_databatch_inplace expects a flat list of
    # tensors when is_preprocessed=True. Use the flat-list convention here.
    return {
        model.input_video_key:   [video],
        "action":                [action],
        "raw_action_dim":        [torch.tensor(raw_action_dim, dtype=torch.long, device=device)],
        "mode":                  ["forward_dynamics"],
        model.input_caption_key: [caption],
        "text_token_ids":        [_tokenize(model, caption, device)],
        "image_size":            [torch.tensor([[h, w, h, w]], dtype=torch.float32, device=device)],
        "fps":              torch.tensor([16.0], device=device),
        "conditioning_fps": torch.tensor([16.0], device=device),
        "num_frames":       torch.tensor([num_video_frames], device=device),
        "domain_id":        [torch.tensor(get_domain_id(domain_name), dtype=torch.long, device=device)],
        "sequence_plan":    [sequence_plan],
        "is_preprocessed":  True,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main loop. Three things only: build batch → training_step → backward+step.
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Local directory containing config.json (architecture only — weights are "
             "randomly initialized). If omitted, fetches Cosmos3-Nano's config.json from HF.",
    )
    parser.add_argument("--num-iters", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path("outputs/trainer_level_training").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build the bare OmniMoTModel (random weights — see module docstring) ---
    model = _load_omni_model(config_dir_arg=args.config_dir)
    model.train()

    print(f"Modality flags: vision_gen={model.config.vision_gen}, "
          f"action_gen={model.config.action_gen}, sound_gen={model.config.sound_gen}")

    # 2) Optimizer — SGD (zero state) so the demo fits on a single 80GB GPU.
    #    Production cosmos_framework training uses AdamW with FSDP across ≥ 8 GPUs.
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
    )

    # 3) Build an alternating multi-modality stream -----------------------
    caption_img = "A neon city street at night, rain reflecting the signs."
    caption_vid = "A camera dollies through a forest of giant glowing mushrooms."
    caption_act = "A robot arm picks up a red block from the table."
    caption_snd = "Wind howling through pine trees, distant thunder."

    def next_batch(it: int):
        # Round-robin through 4 modalities. Replace with your real dataloader.
        kind = ["T2I", "T2V", "ACTION_FDM", "T2VS"][it % 4]
        if kind == "T2I":
            return (kind, make_text_to_image_batch(model, caption=caption_img))
        if kind == "T2V":
            return (kind, make_text_to_video_batch(model, caption=caption_vid))
        if kind == "ACTION_FDM":
            return (kind, make_action_fdm_batch(model, caption=caption_act))
        return (kind, make_sound_video_batch(model, caption=caption_snd))

    # 4) Training loop ----------------------------------------------------
    # model.training_step does, end-to-end:
    #   tokenize text → VAE-encode video → sample t & noise (rectified flow)
    #   → pack tokens → run MoT network → flow-matching velocity loss.
    # We just call it.
    for it in range(args.num_iters):
        kind, batch = next_batch(it)

        aux, loss = model.training_step(batch, iteration=it)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        print(f"iter {it:>3d}  [{kind}]  loss={loss.item():.4f}")

    # 5) Save weights — plain torch.save ----------------------------------
    # NOTE: production cosmos_framework writes sharded DCP via cosmos_framework.utils.checkpoint
    # (FSDP-aware, resumable). torch.save is fine for this single-GPU demo
    # but won't capture FSDP shards or optimizer state.
    save_path = output_dir / "model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights: {save_path}")


if __name__ == "__main__":
    main()
