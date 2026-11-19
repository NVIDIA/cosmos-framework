#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Autoregressive forward-dynamics rollout of held-out LIBERO episodes + per-frame metrics.

For every held-out episode: condition on the REAL first frame and the REAL actions, generate
the episode chunk by chunk (16 actions -> 16 frames), re-conditioning each chunk on the LAST
GENERATED frame (no ground-truth frame after frame 0), and compare with the real recording at
the model canvas (the 160x320 content of the 320x192 canvas; the real frames are resized with
the training transform). Per frame: PSNR, SSIM, LPIPS (whole frame and per view: left =
third-person, right = wrist). Saves real|generated videos, raw frames (npz) for downstream
checks (drift curve, gripper-state classifier) and a summary JSON.

Model loading mirrors cosmos_framework.scripts.action_policy_server_libero (OmniInference on a
DCP iter dir + the run's config.yaml). Prompt/action preprocessing reuses the training dataset
(LIBEROLeRobotDataset._build_result) and ActionTransformPipeline so the inference batch is
built exactly like a training sample (JSON prompt, quantile_rot normalization, padding).
"""

from cosmos_framework.inference.common.init import init_script  # noqa: E402  (must run before torch-heavy imports)

init_script()

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

VIEW_DESC = "The left half shows the third-person view; the right half shows the wrist-mounted camera."
CANVAS_W, CANVAS_H = 320, 192  # "256" tier, 16:9 bucket for the 256x512 concat view


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint-path", required=True, help="DCP checkpoint iter dir, e.g. .../checkpoints/iter_000000250"
    )
    ap.add_argument("--config-file", required=True, help="the training run's config.yaml")
    ap.add_argument("--root", required=True, help="libero_10 LeRobot dir")
    ap.add_argument("--holdout", required=True, help="holdout episode id list")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--episodes", default="", help="comma-separated subset of held-out episode ids (default all)")
    ap.add_argument(
        "--max-chunks", type=int, default=0, help="debug: stop after N chunks per episode (0 = full episode)"
    )
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=30, help="denoising steps (forward_dynamics default 30)")
    ap.add_argument("--guidance", type=float, default=1.0, help="CFG scale (forward_dynamics default 1.0)")
    ap.add_argument("--shift", type=float, default=3.0, help="UniPC time shift; 3.0 = training shift for the 256 tier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true", help="use regular (non-EMA) weights")
    ap.add_argument("--lpips-net", default="alex", choices=["alex", "vgg"])
    return ap.parse_args()


def load_model(ckpt: str, config_file: str, output_dir: Path, use_ema: bool):
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.common.init import init_output_dir
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.scripts.action_policy_server_utils import (
        disable_runtime_ema_for_frozen_config,
        maybe_init_distributed,
    )

    maybe_init_distributed()
    ov = OmniSetupOverrides.model_validate(
        {"checkpoint_path": str(ckpt), "config_file": str(config_file), "use_ema_weights": bool(use_ema)}
    )
    ov.output_dir = output_dir
    ov.sampler = "unipc"
    setup_args = ov.build_setup()
    init_output_dir(setup_args.output_dir)
    setup_args = disable_runtime_ema_for_frozen_config(setup_args)
    pipe = OmniInference.create(setup_args)
    model = pipe.model
    model.eval()
    return pipe, model


def to_canvas_u8(video_u8_cthw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[3,T,256,512] uint8 -> ([3,T,160,320] uint8 content, image_size[4]) via the training resize/pad path."""
    from cosmos_framework.data.generator.action.utils.transforms import (
        reflection_pad_to_target,
        remove_reflection_padding,
    )

    d: dict = {"video": video_u8_cthw}
    reflection_pad_to_target(d, ["video"], True, CANVAS_W, CANVAS_H)
    return remove_reflection_padding(d["video"], d["image_size"]), d["image_size"]


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else float(10.0 * math.log10(255.0**2 / mse))


def write_mp4(path: Path, frames_thwc_u8: np.ndarray, fps: int = 20) -> None:
    import imageio.v2 as imageio

    w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1)
    for f in frames_thwc_u8:
        w.append_data(f)
    w.close()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(exist_ok=True)

    t_load0 = time.time()
    pipe, model = load_model(args.checkpoint_path, args.config_file, out / "_omni_inference", use_ema=not args.no_ema)
    t_load = time.time() - t_load0
    device = torch.device("cuda")

    import lpips
    from skimage.metrics import structural_similarity

    lpips_fn = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()

    from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset
    from cosmos_framework.data.generator.action.utils.action_processing import make_batched_action_processing_fields
    from cosmos_framework.data.generator.action.utils.transforms import (
        ActionTransformPipeline,
        remove_reflection_padding,
    )

    ds = LIBEROLeRobotDataset(
        root=args.root,
        fps=20,
        chunk_length=args.chunk,
        mode="forward_dynamics",
        camera_mode="concat_view",
        image_size=256,
        action_space="frame_wise_relative",
        rotation_space="6d",
        pose_coordinate_frame="native",
        action_normalization="quantile_rot",
        split="val",
        holdout_episodes_path=args.holdout,
    )
    max_action_dim = int(model.config.max_action_dim)
    transform = ActionTransformPipeline(
        tokenizer_config=None,
        cfg_dropout_rate=0.0,
        max_action_dim=max_action_dim,
        append_idle_frames=True,
        idle_frames_dropout=0.0,
        format_prompt_as_json=True,
    )

    ep_ids = [int(e) for e in ds._ep_vals.tolist()]
    if args.episodes:
        want = {int(x) for x in args.episodes.split(",") if x.strip()}
        ep_ids = [e for e in ep_ids if e in want]
    print(f"[eval] episodes: {ep_ids}", flush=True)

    run_cfg = dict(
        checkpoint_path=str(args.checkpoint_path),
        config_file=str(args.config_file),
        use_ema_weights=not args.no_ema,
        num_steps=args.num_steps,
        guidance=args.guidance,
        shift=args.shift,
        seed=args.seed,
        chunk=args.chunk,
        lpips_net=args.lpips_net,
        canvas=[CANVAS_H, CANVAS_W],
        content=[160, 320],
        model_load_s=round(t_load, 1),
    )
    (out / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    per_episode = []
    all_rows = []
    t_eval0 = time.time()
    for ep_i, ep in enumerate(ep_ids):
        rows = np.flatnonzero(ds._row_episode == ep)
        assert len(rows) > 1 and np.all(np.diff(rows) == 1), "episode rows must be contiguous"
        T = int(len(rows))
        ts = [float(ds._row_timestamp[j]) for j in rows]
        episode = ds._episodes[ep]
        task = ds._tasks[int(ds._row_task[rows[0]])].split(" | ")[0].strip()
        t0 = time.time()
        real01 = ds._load_video(episode, ts)  # [T,3,256,512] float [0,1]
        act10 = ds._build_frame_wise_action(ds._row_action[rows])  # [T,10] raw (unnormalized)
        t_decode = time.time() - t0
        real_u8_cthw = (real01 * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3)  # [3,T,256,512]
        real_canvas, _ = to_canvas_u8(real_u8_cthw)  # [3,T,160,320]
        real_canvas = real_canvas.permute(1, 0, 2, 3).contiguous().numpy()  # [T,3,160,320]

        n_chunks = math.ceil((T - 1) / args.chunk)
        if args.max_chunks > 0:
            n_chunks = min(n_chunks, args.max_chunks)
        T_eval = min(T, 1 + n_chunks * args.chunk)
        gen = np.zeros((T_eval, 3, 160, 320), dtype=np.uint8)
        gen[0] = real_canvas[0]
        cond01 = real01[0]  # [3,256,512] float [0,1]; the only ground-truth frame the model ever sees
        chunk_times = []
        for k in range(n_chunks):
            f0 = k * args.chunk
            a = act10[f0 : f0 + args.chunk]
            n_new = min(args.chunk, T - 1 - f0)
            if a.shape[0] < args.chunk:  # last partial chunk: repeat the final action
                a = torch.cat([a, a[-1:].repeat(args.chunk - a.shape[0], 1)], dim=0)
            video_in = cond01.unsqueeze(0).repeat(args.chunk + 1, 1, 1, 1)  # [17,3,H,W]; frames 1..16 are noised anyway
            item = ds._build_result(
                mode="forward_dynamics",
                video=video_in,
                action=a,
                ai_caption=task,
                additional_view_description=VIEW_DESC,
            )
            item = transform(item, resolution=None)
            cap = item["ai_caption"]
            cap = json.dumps(cap) if isinstance(cap, dict) else str(cap)
            batch = {
                model.input_video_key: [[item["video"]]],  # uint8 [3,17,192,320]
                "action": [[item["action"]]],  # [16,64] normalized + padded
                **make_batched_action_processing_fields(item["action_processing_record"], 1),
                "mode": ["forward_dynamics"],
                "ai_caption": [cap],
                "prompt": [cap],
                "conditioning_fps": [item["conditioning_fps"]],
                "image_size": item["image_size"].unsqueeze(0).to(device),
                "domain_id": [item["domain_id"]],
                "sequence_plan": [item["sequence_plan"]],
            }
            seed_k = args.seed * 100000 + ep * 100 + k
            tc0 = time.time()
            with torch.inference_mode():
                samples = model.generate_samples_from_batch(
                    batch,
                    guidance=args.guidance,
                    seed=[seed_k],
                    num_steps=args.num_steps,
                    shift=args.shift,
                    has_negative_prompt=False,
                )
                vid = model.decode(samples["vision"][0]).squeeze(0)  # [3,17,192,320] in [-1,1]
            vid = remove_reflection_padding(vid, item["image_size"])  # [3,17,160,320]
            u8 = ((vid.float().clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
            chunk_times.append(time.time() - tc0)
            gen[f0 + 1 : f0 + 1 + n_new] = u8[:, 1 : 1 + n_new].permute(1, 0, 2, 3).numpy()
            cond01 = u8[:, args.chunk].float() / 255.0  # last generated frame -> next conditioning frame
            if k == 0 and ep_i == 0:
                print(f"[eval] prompt example: {cap}", flush=True)
            print(f"[eval] ep {ep} chunk {k + 1}/{n_chunks} {chunk_times[-1]:.1f}s", flush=True)

        # ---- metrics (frames 1..T_eval-1; frame 0 is the given conditioning frame) ----
        real_e = real_canvas[:T_eval]
        with torch.no_grad():
            xr = torch.from_numpy(real_e).to(device).float() / 127.5 - 1.0
            xg = torch.from_numpy(gen).to(device).float() / 127.5 - 1.0
            lp = lpips_fn(xr, xg).flatten().cpu().numpy()
            lp_l = lpips_fn(xr[..., :160], xg[..., :160]).flatten().cpu().numpy()
            lp_r = lpips_fn(xr[..., 160:], xg[..., 160:]).flatten().cpu().numpy()
        ep_rows = []
        for t in range(1, T_eval):
            r_hwc = real_e[t].transpose(1, 2, 0)
            g_hwc = gen[t].transpose(1, 2, 0)
            row = dict(
                episode=ep,
                task=task,
                frame=t,
                chunk=(t - 1) // args.chunk,
                pos_in_chunk=(t - 1) % args.chunk + 1,
                psnr=psnr_u8(real_e[t], gen[t]),
                ssim=float(structural_similarity(r_hwc, g_hwc, channel_axis=2, data_range=255)),
                lpips=float(lp[t]),
                psnr_left=psnr_u8(real_e[t][..., :160], gen[t][..., :160]),
                psnr_right=psnr_u8(real_e[t][..., 160:], gen[t][..., 160:]),
                ssim_left=float(structural_similarity(r_hwc[:, :160], g_hwc[:, :160], channel_axis=2, data_range=255)),
                ssim_right=float(structural_similarity(r_hwc[:, 160:], g_hwc[:, 160:], channel_axis=2, data_range=255)),
                lpips_left=float(lp_l[t]),
                lpips_right=float(lp_r[t]),
                gripper_cmd=float(act10[t - 1, 9].item()),
            )
            ep_rows.append(row)
        all_rows.extend(ep_rows)

        ed = out / "episodes" / f"ep{ep:03d}"
        ed.mkdir(exist_ok=True)
        np.savez_compressed(
            ed / "frames.npz",
            gen=gen,
            real=real_e,
            actions10=act10[:T_eval].numpy(),
            gripper_state=ds._row_action[rows][:T_eval, 6],
        )
        write_mp4(ed / "gen.mp4", gen.transpose(0, 2, 3, 1))
        write_mp4(ed / "real.mp4", real_e.transpose(0, 2, 3, 1))
        divider = np.full((T_eval, 160, 4, 3), 255, dtype=np.uint8)
        sbs = np.concatenate(
            [real_e.transpose(0, 2, 3, 1), divider, gen.transpose(0, 2, 3, 1)], axis=2
        )  # [T,160,644,3]
        write_mp4(ed / "side_by_side.mp4", sbs)
        with (ed / "metrics.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ep_rows[0].keys()))
            w.writeheader()
            w.writerows(ep_rows)
        m = {
            k: float(np.mean([r[k] for r in ep_rows]))
            for k in (
                "psnr",
                "ssim",
                "lpips",
                "psnr_left",
                "psnr_right",
                "ssim_left",
                "ssim_right",
                "lpips_left",
                "lpips_right",
            )
        }
        ep_sum = dict(
            episode=ep,
            task=task,
            frames_total=T,
            frames_evaluated=T_eval - 1,
            chunks=n_chunks,
            gen_s_per_chunk=float(np.mean(chunk_times)),
            gen_s_total=float(np.sum(chunk_times)),
            decode_s=round(t_decode, 2),
            **m,
        )
        (ed / "meta.json").write_text(json.dumps(ep_sum, indent=2))
        per_episode.append(ep_sum)
        print(
            f"[eval] ep {ep} done: PSNR {m['psnr']:.2f} SSIM {m['ssim']:.3f} LPIPS {m['lpips']:.3f} "
            f"({T_eval - 1} frames, {np.sum(chunk_times):.0f}s)",
            flush=True,
        )

    # ---- aggregate ----
    with (out / "per_frame_all.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    keys = ("psnr", "ssim", "lpips", "psnr_left", "psnr_right", "ssim_left", "ssim_right", "lpips_left", "lpips_right")
    overall = {k: float(np.mean([r[k] for r in all_rows])) for k in keys}
    overall_median = {k: float(np.median([r[k] for r in all_rows])) for k in keys}
    by_chunk = {}
    for c in sorted({r["chunk"] for r in all_rows}):
        rs = [r for r in all_rows if r["chunk"] == c]
        by_chunk[str(c)] = {
            "n_frames": len(rs),
            "n_episodes": len({r["episode"] for r in rs}),
            **{k: float(np.mean([r[k] for r in rs])) for k in ("psnr", "ssim", "lpips")},
        }
    by_pos = {}
    for p in range(1, args.chunk + 1):
        rs = [r for r in all_rows if r["pos_in_chunk"] == p]
        if rs:
            by_pos[str(p)] = {k: float(np.mean([r[k] for r in rs])) for k in ("psnr", "ssim", "lpips")}
    summary = dict(
        run_config=run_cfg,
        n_episodes=len(per_episode),
        n_frames_evaluated=len(all_rows),
        overall_mean=overall,
        overall_median=overall_median,
        by_chunk_index=by_chunk,
        by_position_in_chunk=by_pos,
        per_episode=per_episode,
        wall_s_total=round(time.time() - t_eval0, 1),
        gen_s_per_chunk_mean=float(np.mean([e["gen_s_per_chunk"] for e in per_episode])),
        peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "n_episodes",
                    "n_frames_evaluated",
                    "overall_mean",
                    "wall_s_total",
                    "gen_s_per_chunk_mean",
                    "peak_gpu_mem_gb",
                )
            },
            indent=1,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
