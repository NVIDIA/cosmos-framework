# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Minimal inference demo — drive Cosmos's OmniMoTModel directly.

⚠  THIS IS A WIRING DEMO. It shows the smallest worked example of the inference
   call sequence for each generation mode, not a production serving recipe.
   For batched / streaming / Ray-Serve deployment, see
   `cosmos_framework.inference.inference.OmniInference` and `cosmos_framework.inference.ray.*`.

⚠  For `--mode action_fdm` and `--mode t2vs` we feed the model RANDOM
   conditioning tensors (no real video / action files on disk), so the
   *output* is just visual / audio noise. The wiring is what's being
   demonstrated — for meaningful outputs you must supply real conditioning
   data via your own loader.

================================================================================
SCOPE
================================================================================
This is NOT "extracting the model into another framework". The cosmos_framework package
must be installed. What this script demonstrates is the smallest possible
inference path per generation mode: load → batch → generate → decode → save.

What we USE from cosmos_framework:
    cosmos_framework.inference.model.Cosmos3OmniModel          → checkpoint loader
    cosmos_framework.inference.common.init.init_script         → 1-line torch.distributed init
    cosmos_framework.inference.{args,inference}                → OmniSampleOverrides +
                                                       get_sample_data (T2I/T2V only)
    cosmos_framework.data.vfm.{action,sequence_packing}        → SequencePlan helpers (action/sound)
    cosmos_framework.model.vfm.vlm.qwen3_vl.utils.tokenize_caption
    model.generate_samples_from_batch(batch, seed)   → THE inference call (CFG + sampler)
    model.decode(latent)                             → VAE decode

What we DO NOT use:
    cosmos_framework.scripts.inference                         → CLI entry point
    cosmos_framework.inference.inference.OmniInference         → serving/batching pipeline
    cosmos_framework.inference.ray.*                           → Ray serving

================================================================================
RUN
================================================================================
    PYTHONPATH=. python examples/integration/trainer_level_inference.py                              # T2I
    PYTHONPATH=. python examples/integration/trainer_level_inference.py --mode t2v                   # T2V
    PYTHONPATH=. python examples/integration/trainer_level_inference.py --mode action_fdm            # action (fake input)
    PYTHONPATH=. python examples/integration/trainer_level_inference.py --mode t2vs                  # sound+video (fake input)
"""

from cosmos_framework.inference.common.init import init_script

init_script()  # init torch.distributed + DCP wrappers (required even on 1 GPU)

import argparse
from pathlib import Path

import torch

from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.inference.args import DEFAULT_CHECKPOINT, OmniSampleOverrides
from cosmos_framework.inference.inference import get_sample_data
from cosmos_framework.inference.model import Cosmos3OmniModel
from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import tokenize_caption
from cosmos_framework.tools.visualize.video import save_img_or_video


# ────────────────────────────────────────────────────────────────────────────
# Per-mode batch builders. T2I and T2V reuse cosmos_framework's `get_sample_data` helper
# (which also stamps default sampler args). action_fdm and t2vs are built by
# hand using the same dict contract as trainer_level_training.py.
# ────────────────────────────────────────────────────────────────────────────

def _tokenize(model, caption: str, device) -> torch.Tensor:
    ids = tokenize_caption(caption, model.vlm_tokenizer, is_video=False,
                           use_system_prompt=model.vlm_config.use_system_prompt)
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def build_t2iv_batch(model, output_dir, prompt: str, num_frames: int) -> dict:
    """T2I (num_frames=1) or T2V (num_frames>1) via cosmos_framework's inference batch helper."""
    sample_args = OmniSampleOverrides(
        name="integration_demo", output_dir=output_dir,
        prompt=prompt, num_frames=num_frames,
    ).build_sample(model_config=model.config)
    return get_sample_data(sample_args, model)


def build_action_fdm_batch(model, *, caption: str, num_video_frames: int = 5,
                           action_chunk: int = 4, raw_action_dim: int = 7,
                           h: int = 128, w: int = 128,
                           domain_name: str = "bridge_orig_lerobot", device="cuda") -> dict:
    """Forward-dynamics inference batch (RANDOM video + actions; output = noise)."""
    video = (torch.randn(1, 3, num_video_frames, h, w, device=device) * 0.3).clamp(-1, 1)
    action = torch.zeros(action_chunk, model.config.max_action_dim, device=device)
    action[:, :raw_action_dim] = torch.randn(action_chunk, raw_action_dim, device=device) * 0.1
    sequence_plan = build_sequence_plan_from_mode(
        mode="forward_dynamics", video_length=num_video_frames,
        action_length=action_chunk, has_text=True,
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


def build_t2vs_batch(model, *, caption: str, num_video_frames: int = 5,
                     audio_hop_count: int = 8, h: int = 128, w: int = 128,
                     device="cuda") -> dict:
    """Text→video+sound inference batch (RANDOM conditioning; output = noise)."""
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


# ────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Local DCP checkpoint dir. If omitted, downloads the default checkpoint.")
    parser.add_argument("--mode", type=str, default="t2i",
                        choices=["t2i", "t2v", "action_fdm", "t2vs"],
                        help="Generation mode. action_fdm and t2vs use random conditioning → noise output.")
    parser.add_argument("--prompt", type=str,
                        default="A neon city street at night, rain reflecting the signs.")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Number of video frames. Defaults: 1 for t2i, 33 for t2v, 5 for action/sound.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=35,
                        help="Sampling steps. Lower → faster + noisier.")
    args = parser.parse_args()

    output_dir = Path(f"outputs/trainer_level_inference/{args.mode}").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load the bare OmniMoTModel ----------------------------------------
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(DEFAULT_CHECKPOINT.download())
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = Cosmos3OmniModel.from_pretrained_dcp(
        checkpoint_path,
        parallelism_config=ParallelismConfig(use_torch_compile=False),
    ).model
    model.eval()

    # 2) Build a batch per mode --------------------------------------------
    if args.mode == "t2i":
        nframes = args.num_frames if args.num_frames is not None else 1
        data_batch = build_t2iv_batch(model, output_dir, args.prompt, nframes)
    elif args.mode == "t2v":
        nframes = args.num_frames if args.num_frames is not None else 33
        data_batch = build_t2iv_batch(model, output_dir, args.prompt, nframes)
    elif args.mode == "action_fdm":
        nframes = args.num_frames if args.num_frames is not None else 5
        data_batch = build_action_fdm_batch(model, caption=args.prompt, num_video_frames=nframes)
    elif args.mode == "t2vs":
        nframes = args.num_frames if args.num_frames is not None else 5
        data_batch = build_t2vs_batch(model, caption=args.prompt, num_video_frames=nframes)
    print(f"Mode: {args.mode}  num_frames={nframes}")

    # 3) Generate. THE only model call needed ------------------------------
    with torch.no_grad():
        outputs = model.generate_samples_from_batch(
            data_batch, seed=[args.seed], num_steps=args.num_steps,
        )

    # 4) Decode vision (and sound if present) ------------------------------
    pixels = model.decode(outputs["vision"][0])              # [1, 3, T, H, W] in [-1, 1]
    pixels = (pixels.clamp(-1, 1) + 1.0) / 2.0               # → [0, 1]

    fps = float(data_batch["fps"][0].item())
    save_img_or_video(pixels[0], str(output_dir / "output"), fps=fps)

    if args.mode == "t2vs" and "sound" in outputs and outputs["sound"] is not None:
        # Sound latents → waveform via AVAE decode. Save as a raw .pt; users plug
        # their own audio writer (torchaudio.save / soundfile) for .wav output.
        sound_latent = outputs["sound"][0]                    # [C_sound, T_sound]
        waveform = model.decode_sound(sound_latent)           # [C_audio, N_samples]
        torch.save(waveform.cpu(), output_dir / "sound.pt")
        print(f"  sound waveform: shape={tuple(waveform.shape)} → sound.pt")

    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
