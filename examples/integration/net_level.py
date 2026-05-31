# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
"net-only" demo: drive `Cosmos3VFMNetwork` (= `model.net`) directly.

================================================================================
⚠  THIS IS A WIRING DEMO, NOT A PRODUCTION RECIPE
================================================================================
The code below shows HOW to call `net.forward` and where to put your loss /
sampler — it does NOT reproduce cosmos_framework's training or sampling recipe.
Concretely, this demo deliberately simplifies four things:

    1. WEIGHTS — `model.net` is RANDOM-INITIALIZED. The demo never loads the
       ~30 GB Cosmos3-Nano DCP shards, so losses and samples are meaningless;
       the point is to show the call sequence. For real weight loading see
       `cosmos_framework.inference.model.Cosmos3OmniModel.from_pretrained_dcp`.
    2. NOISE SCHEDULE — uses a fixed σ = 0.5 every iter.
       Real training samples σ from a logit-normal (image) / waver (video)
       distribution per `OmniMoTModel._get_train_noise_level_vision`.
    3. LOSS — plain MSE on velocity.
       Real training uses `cosmos_framework.model.vfm.algorithm.loss.flow_matching
       .compute_flow_matching_loss`, which adds per-sample weighting,
       condition-mask zeroing, and `loss_scale=10` (with separate image/video
       scaling).
    4. SAMPLER — plain Euler, no CFG, ~8 steps.
       Real inference uses UniPC (`cosmos_framework.model.vfm.diffusion.samplers.unipc`)
       with `guidance=1.5` and 35 steps.

If you train or sample with the demo's simplifications you will diverge from
the Cosmos3 recipe. Use this file to learn the API surface, then swap in the
real weights + loss + sampler when porting to your own framework.

================================================================================
WHAT THIS SHOWS
================================================================================
The previous demos (trainer_level_inference.py / trainer_level_training.py) call
`OmniMoTModel.generate_samples_from_batch` and `OmniMoTModel.training_step`,
which are 2000+ line orchestration methods.

This demo goes one level deeper: it extracts the *core denoiser network*

    net = model.net          # type: Cosmos3VFMNetwork (a plain nn.Module)

and calls `net.forward(packed_seq, fps_vision=...)` itself, then writes the
flow-matching loss and the sampling loop by hand. The point: `net` is the
unit you would port into another training framework — the surrounding
`OmniMoTModel` is just orchestration around it.

================================================================================
THE 3 LAYERS
================================================================================
  ┌────────────────────────────────────────────────────────────────────────────
  │ OmniMoTModel  (model)                                                    │
  │   ├── training_step(), generate_samples_from_batch()  ← orchestration    │
  │   ├── encode/decode = VAE                             ← cosmos_framework VAE       │
  │   ├── _pack_input_sequence(...)                       ← cosmos_framework packer    │
  │   │                                                                       │
  │   └── net = Cosmos3VFMNetwork  ◀──── THIS DEMO'S FOCUS                    │
  │           forward(packed_seq, fps_vision=...) -> {"preds_vision": ...}    │
  └────────────────────────────────────────────────────────────────────────────

`net`'s I/O contract:
  INPUT  : a `PackedSequence` (text tokens + noised vision latents +
           attention modes + position ids), built via
           `model._pack_input_sequence(...)`.
  OUTPUT : dict with `preds_vision` = list of velocity tensors [C, T, H, W],
           one per sample.

================================================================================
WHAT IS STILL "COSMOS" IN THIS DEMO
================================================================================
To USE `net`, you still need to BUILD its input. We use these cosmos_framework helpers
to do that — they are unavoidable unless you re-implement the packer:

    model.encode(...) / model.decode(...)        ← VAE (pixels ↔ latents)
    model._load_and_tokenize_text_data(...)      ← text tokenization
    build_sequence_plans_from_data_batch(...)    ← per-sample modality plan
    model._pack_input_sequence(...)              ← builds the PackedSequence
    model._add_noise_to_input(...)               ← rectified-flow noising
    model._replace_clean_with_noised(...)        ← splice xt into packed_seq
    model._prepare_inference_data(...)           ← inference-time data prep
    model._get_velocity(net=net, ...)            ← inference per-step helper
                                                    (just packs + calls net)

If you wanted ZERO cosmos_framework imports at runtime, you would re-vendor
`cosmos_framework/data/vfm/sequence_packing.py` and the VAE into your own framework.

================================================================================
RUN
================================================================================
    PYTHONPATH=. python examples/integration/net_level.py
    PYTHONPATH=. python examples/integration/net_level.py --config-dir /path/to/dir/with/config.json
"""

from cosmos_framework.inference.common.init import init_script

init_script(training=True)

import argparse
import json
from pathlib import Path

import attrs
import torch
import torch.nn.functional as F

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.data.vfm.sequence_packing import SequencePlan, build_sequence_plans_from_data_batch
from cosmos_framework.inference.args import DEFAULT_CHECKPOINT
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import tokenize_caption


def _load_omni_model(*, config_dir_arg: str | None):
    """Build OmniMoTModel with RANDOM main-transformer weights — wiring demo only.

    This helper exists so the demo can run without downloading the ~30 GB transformer
    DCP. Only ``config.json`` is fetched (single ~5 KB file) and the main net is
    instantiated via ``hydra.utils.instantiate`` with random parameters. Auxiliary
    sub-models (Qwen3-VL tokenizer, Wan2.2 VAE, AVAE) still load from the HF cache
    during ``Cosmos3OmniModel.__init__`` — they are not stubbed out.

    For REAL weight loading, see
    :func:`cosmos_framework.inference.model.Cosmos3OmniModel.from_pretrained_dcp`.
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
        ("cosmos3._src.vfm.models.", "cosmos_framework.model.vfm."),
        ("cosmos3._src.vfm.tokenizers.", "cosmos_framework.model.vfm.tokenizers."),
        ("cosmos3._src.imaginaire.", "cosmos_framework."),
    ]:
        config_text = config_text.replace(_old, _new)
    config = Cosmos3OmniConfig(model=json.loads(config_text)["model"])
    config.parallelism = attrs.asdict(ParallelismConfig())
    config.compile = attrs.asdict(CompileConfig(enabled=False))
    return Cosmos3OmniModel(config).model


# ────────────────────────────────────────────────────────────────────────────
# Batch builders — same shapes as trainer_level_training.py uses.
# ────────────────────────────────────────────────────────────────────────────
def _tokenize(model, caption: str, device) -> torch.Tensor:
    ids = tokenize_caption(
        caption, model.vlm_tokenizer,
        is_video=False, use_system_prompt=model.vlm_config.use_system_prompt,
    )
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, N_tok]


def make_text_to_image_batch(model, *, caption: str, h: int = 128, w: int = 128, device="cuda") -> dict:
    image = (torch.randn(1, 3, 1, h, w, device=device) * 0.3).clamp(-1, 1)
    return {
        model.input_image_key:   [image],
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
    """Joint text→video+sound (t2vs). See trainer_level_training.py for full contract."""
    waveform = (torch.randn(2, audio_hop_count * 1920, device=device) * 0.1).clamp(-1, 1)
    video = (torch.randn(1, 3, num_video_frames, h, w, device=device) * 0.3).clamp(-1, 1)
    sequence_plan = SequencePlan(has_text=True, has_vision=True, has_sound=True)
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
    """Forward-dynamics action batch. See trainer_level_training.py for the contract."""
    video = (torch.randn(1, 3, num_video_frames, h, w, device=device) * 0.3).clamp(-1, 1)
    action = torch.zeros(action_chunk, model.config.max_action_dim, device=device)
    action[:, :raw_action_dim] = torch.randn(action_chunk, raw_action_dim, device=device) * 0.1
    sequence_plan = build_sequence_plan_from_mode(
        mode="forward_dynamics",
        video_length=num_video_frames,
        action_length=action_chunk,
        has_text=True,
    )
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


# ════════════════════════════════════════════════════════════════════════════
# TRAINING: forward + backward through `net` only, custom flow-matching loss.
# ════════════════════════════════════════════════════════════════════════════
def train_one_step(model, net, batch, *, iteration: int) -> torch.Tensor:
    """One rectified-flow training step using only `net.forward`.

    Equivalent to a single `model.training_step(batch, iteration)` call, but
    with the network forward, the loss, and the backward all written here so
    you can see — and replace — each piece.
    """
    # ── 1. Build the input contract for `net`. These calls are all cosmos_framework
    #       preprocessing — they go away if you re-implement the packer.
    input_text_indexes = model._load_and_tokenize_text_data(batch, iteration)
    sequence_plans = build_sequence_plans_from_data_batch(
        data_batch=batch,
        input_video_key=model.input_video_key,
        input_image_key=model.input_image_key,
    )
    gen_data_clean = model.get_data_and_condition(batch, iteration=iteration)

    # Pick a mid-range noise level for the demo (real training samples sigma
    # from a per-modality distribution; see cosmos_framework.model.vfm.omni_mot_model
    # `_get_train_noise_level_vision`).
    B = gen_data_clean.batch_size
    # tensor_kwargs_fp32 = {"dtype": float32, "device": ...} — keeps demo
    # tensors on the same device / dtype the model expects.
    sigmas = torch.full((B, 1), 0.5, **model.tensor_kwargs_fp32)        # [B, 1]
    timesteps = (sigmas * 1000.0).cpu()                                 # [B, 1] on cpu

    packed_seq = model._pack_input_sequence(
        sequence_plans, input_text_indexes, gen_data_clean, timesteps,
    )
    gen_data_noised = model._add_noise_to_input(
        gen_data_clean, packed_seq, sigmas, iteration=iteration,
    )
    model._replace_clean_with_noised(packed_seq, gen_data_noised)
    packed_seq.to_cuda()

    # ── 2. THE bare-net forward pass. This is the single line that you would
    #       call from your own training loop after porting `net` into it.
    out = net(packed_seq, fps_vision=gen_data_clean.fps_vision)         # type: dict
    v_pred = out["preds_vision"]                                        # list of [C, T, H, W]

    # ── 3. Custom flow-matching loss (MSE on velocity). This is what
    #       `cosmos_framework.model.vfm.algorithm.loss.flow_matching.compute_flow_matching_loss`
    #       computes, minus the per-sample weighting & condition masking.
    v_target = gen_data_noised.vt_target_vision                         # list of [C, T, H, W]
    loss = sum(F.mse_loss(p.float(), t.float()) for p, t in zip(v_pred, v_target))

    # ── 4. Backward. Your code, not cosmos_framework's.
    loss.backward()
    return loss


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE: hand-written Euler sampling loop, each step a `net.forward`.
# Generic across modalities: returns whatever the batch's sequence plans say
# is present (vision always; action and/or sound when configured).
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def sample(model, net, batch, *, num_steps: int = 12) -> dict:
    """N-step Euler integration of dx/dt = v(x,t) — no cosmos_framework sampler involved.

    Production cosmos_framework uses UniPC/EDM (cosmos_framework.model.vfm.diffusion.samplers.*).
    Plain Euler keeps the loop on one screen and surfaces where `net` is called.

    Returns a dict:
        "pixels":         Tensor[3, T, H, W] in [0,1] — always present
        "action":         Tensor[T_action, action_dim] — only if has_action
        "sound_waveform": Tensor[C_audio, N_samples]    — only if has_sound

    For ACTION_FDM and T2VS the demo's batch uses random conditioning, so
    these outputs are noise. The wiring is what's being demonstrated.
    """
    # `_prepare_inference_data` tokenizes cond+uncond captions, builds the per-
    # sample `sequence_plans`, and constructs the initial noise tensor. The
    # noise layout per sample (flat [D]) is concatenation of:
    #     [vision_flat | action_flat (if has_action) | sound_flat (if has_sound)]
    # — same layout `_get_velocity` consumes.
    sequence_plans, gen_data_clean, cond_tokens, _, initial_noise = model._prepare_inference_data(
        batch, seed=[0], has_negative_prompt=False,
    )

    xt = initial_noise                                                   # list[B=1] of flat [D]
    dt = -1.0 / num_steps  # integrate from t=1 (noise) → t=0 (clean)

    for step in range(num_steps):
        t_now = 1.0 - step / num_steps
        timestep = torch.tensor([[t_now * 1000.0]], **model.tensor_kwargs_fp32)  # [1, 1]

        # `_get_velocity` (a) reshapes `xt` back to per-modality tokens,
        # (b) packs them into a PackedSequence with the current timestep,
        # (c) calls `net(packed_seq, ...)`, (d) returns flat velocity.
        v = model._get_velocity(
            net=net,
            noise_x=xt,
            timestep=timestep,
            text_tokens=cond_tokens,
            sequence_plans=sequence_plans,
            gen_data_clean=gen_data_clean,
        )
        xt = [x + dt * v_i for x, v_i in zip(xt, v)]

    # Split the final flat trajectory back into per-modality tensors.
    # Offsets exactly mirror `_get_velocity`'s split.
    has_action = model.config.action_gen and any(p.has_action for p in sequence_plans)
    has_sound  = model.config.sound_gen and any(p.has_sound for p in sequence_plans)

    flat = xt[0]
    offset = 0

    vision_shape = gen_data_clean.x0_tokens_vision[0].shape               # [1, C, T, H, W]
    vision_dim = int(torch.tensor(vision_shape).prod())
    vision_latent = flat[offset : offset + vision_dim].reshape(vision_shape)
    offset += vision_dim

    out: dict = {}
    pixels = model.decode(vision_latent)                                  # [1, 3, T, H, W] in [-1, 1]
    out["pixels"] = (pixels[0].clamp(-1, 1) + 1.0) / 2.0                  # [3, T, H, W] in [0, 1]

    if has_action and gen_data_clean.x0_tokens_action is not None:
        action_shape = gen_data_clean.x0_tokens_action[0].shape           # [T_action, action_dim]
        action_dim = int(torch.tensor(action_shape).prod())
        out["action"] = flat[offset : offset + action_dim].reshape(action_shape)
        offset += action_dim

    if has_sound and gen_data_clean.x0_tokens_sound is not None and sequence_plans[0].has_sound:
        sound_shape = gen_data_clean.x0_tokens_sound[0].shape             # [C_sound, T_sound]
        sound_dim = int(torch.tensor(sound_shape).prod())
        sound_latent = flat[offset : offset + sound_dim].reshape(sound_shape)
        offset += sound_dim
        out["sound_waveform"] = model.decode_sound(sound_latent)          # [C_audio, N_samples]

    return out


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default=None,
                        help="Local directory containing config.json (architecture only — weights are "
                             "randomly initialized). If omitted, fetches Cosmos3-Nano's config.json from HF.")
    parser.add_argument("--num-train-iters", type=int, default=2)
    parser.add_argument("--num-sample-steps", type=int, default=12)
    parser.add_argument("--sample-mode", type=str, default="t2i",
                        choices=["t2i", "t2v", "action_fdm", "t2vs"],
                        help="Which modality to sample. action_fdm/t2vs use random conditioning → noise output.")
    parser.add_argument("--skip-sample", action="store_true",
                        help="Skip the inference section (saves ~1 min).")
    args = parser.parse_args()

    output_dir = Path("outputs/net_level").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build OmniMoTModel (random weights — see module docstring) + grab the bare net
    model = _load_omni_model(config_dir_arg=args.config_dir)

    net = model.net  # ← Cosmos3VFMNetwork; THIS is the unit you'd port
    print(f"net type:   {type(net).__name__}")
    print(f"net params: {sum(p.numel() for p in net.parameters()) / 1e9:.2f} B")

    # 2) TRAINING — forward + backward through `net` ------------------------
    net.train()
    optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=1e-5)

    caption_img = "A neon city street at night, rain reflecting the signs."
    caption_vid = "A camera dollies through a forest of giant glowing mushrooms."
    caption_act = "A robot arm picks up a red block from the table."
    caption_snd = "Wind howling through pine trees, distant thunder."

    def next_batch(it: int):
        kind = ["T2I", "T2V", "ACTION_FDM", "T2VS"][it % 4]
        if kind == "T2I":
            return (kind, make_text_to_image_batch(model, caption=caption_img))
        if kind == "T2V":
            return (kind, make_text_to_video_batch(model, caption=caption_vid))
        if kind == "ACTION_FDM":
            return (kind, make_action_fdm_batch(model, caption=caption_act))
        return (kind, make_sound_video_batch(model, caption=caption_snd))

    for it in range(args.num_train_iters):
        kind, batch = next_batch(it)
        loss = train_one_step(model, net, batch, iteration=it)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"[train] iter {it:>3d}  [{kind}]  loss={loss.item():.4f}")

    # 3) INFERENCE — sampling loop where each step is a `net.forward` -------
    if not args.skip_sample:
        net.eval()

        sample_caption = {
            "t2i":        "A robot standing on a rooftop at sunset.",
            "t2v":        "A camera dollies through a forest of giant glowing mushrooms.",
            "action_fdm": "A robot arm picks up a red block from the table.",
            "t2vs":       "Wind howling through pine trees, distant thunder.",
        }[args.sample_mode]
        sample_builder = {
            "t2i":        lambda: make_text_to_image_batch(model, caption=sample_caption),
            "t2v":        lambda: make_text_to_video_batch(model, caption=sample_caption),
            "action_fdm": lambda: make_action_fdm_batch(model, caption=sample_caption),
            "t2vs":       lambda: make_sound_video_batch(model, caption=sample_caption),
        }[args.sample_mode]

        out = sample(model, net, sample_builder(), num_steps=args.num_sample_steps)

        from cosmos_framework.tools.visualize.video import save_img_or_video
        sample_dir = output_dir / f"sample_{args.sample_mode}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_img_or_video(out["pixels"], str(sample_dir / "output"), fps=16.0)
        print(f"[infer] {args.sample_mode}: pixels saved to {sample_dir / 'output'}")

        if "sound_waveform" in out:
            torch.save(out["sound_waveform"].cpu(), sample_dir / "sound.pt")
            print(f"[infer] {args.sample_mode}: sound shape={tuple(out['sound_waveform'].shape)} → sound.pt")
        if "action" in out:
            torch.save(out["action"].cpu(), sample_dir / "action.pt")
            print(f"[infer] {args.sample_mode}: action shape={tuple(out['action'].shape)} → action.pt")


if __name__ == "__main__":
    main()
