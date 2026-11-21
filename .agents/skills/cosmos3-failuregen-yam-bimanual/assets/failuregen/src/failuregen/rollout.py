# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Render one planned run: grounded prefix, then the injected failure.

A run is a few chunks of the host episode replayed untouched, so the clip establishes a normal
grasp and carry, followed by chunks driven by the spliced trajectory. Each chunk's last frame
seeds the next, so the rollout is autoregressive and the failure has to survive being fed back
in rather than only appearing in one chunk.

Every run writes ``record.json`` next to its clip. That file is the only reason results stay
interpretable weeks later: it carries the donor, the injection instant, the channels taken, the
arm that was holding, and the pose distance.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

from . import annotations as ann
from .config import Profile
from .splice import continuous, rechunk, splice, windows


def concat_frame(profile: Profile, collection: str, episode: int, t: float) -> Image.Image:
    """The tiled multi-camera frame the model conditions on: wide view over the wrist views."""
    import torch
    import torch.nn.functional as F
    from lerobot.datasets.video_utils import decode_video_frames

    tol = 1.0 / profile.dataset.source_fps
    cams = profile.cameras
    front = decode_video_frames(
        str(ann.video_path(profile, collection, episode, cams.front)), [float(t)], tol, backend="pyav"
    )[0]
    if not cams.wrist:
        arr = (front.clamp(0, 1) * 255).to(torch.uint8)
        return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())

    _, h, w = front.shape
    n = len(cams.wrist)
    tile_h = int(round(h * (1.0 - cams.front_fraction) / cams.front_fraction))
    tile_w = w // n
    strip = []
    for _, key in sorted(cams.wrist.items()):
        img = decode_video_frames(
            str(ann.video_path(profile, collection, episode, key)), [float(t)], tol, backend="pyav"
        )[0]
        strip.append(F.interpolate(img.unsqueeze(0), size=(tile_h, tile_w), mode="bilinear", align_corners=False)[0])
    tile = torch.cat([front, torch.cat(strip, dim=-1)], dim=-2)
    return Image.fromarray((tile.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())


def last_frame(mp4: Path, out_png: Path) -> None:
    last = None
    with av.open(str(mp4)) as c:
        for frame in c.decode(c.streams.video[0]):
            last = frame
    if last is None:
        raise RuntimeError(f"no frames in {mp4}")
    last.to_image().save(out_png, format="PNG")


def stitch(clips: list[Path], out_mp4: Path, fps: int) -> None:
    """Join chunk clips, dropping each chunk's first frame after the first.

    Chunk k's opening frame is chunk k-1's closing frame; keeping both makes every boundary
    stutter, which reads as the arm hesitating.
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tb = Fraction(1, fps)
    with av.open(str(out_mp4), "w") as out:
        stream, index = None, 0
        for i, clip in enumerate(clips):
            with av.open(str(clip)) as src:
                src_stream = src.streams.video[0]
                if stream is None:
                    stream = out.add_stream("h264", rate=fps)
                    stream.width = src_stream.codec_context.width
                    stream.height = src_stream.codec_context.height
                    stream.pix_fmt = "yuv420p"
                    stream.options = {"crf": "18"}
                    stream.codec_context.time_base = tb
                for j, frame in enumerate(src.decode(src_stream)):
                    if i > 0 and j == 0:
                        continue
                    frame.pts, frame.time_base = index, tb
                    index += 1
                    for packet in stream.encode(frame):
                        out.mux(packet)
        if stream is not None:
            for packet in stream.encode(None):
                out.mux(packet)


def render(
    profile: Profile, model, run: dict, out_dir: Path, donor: dict | None, seed: int = 0, is_rank0=lambda: True
) -> Path:
    """Render a single run described by a manifest entry."""
    if "prepared" in run:
        from .prepared import render_prepared

        return render_prepared(profile, model, run, out_dir, seed=seed, is_rank0=is_rank0, donor=donor)
    dp = profile.dataset
    chunk = dp.chunk_length
    collection = run.get("collection") or next(iter(dp.roots))
    host_ep = int(run["host_episode"])
    pre, post = int(run["pre_chunks"]), int(run["post_chunks"])
    t_inject = float(run["inject_at_s"])
    mode = run.get("mode", "transplant")
    # Checked before any I/O: an unresolved donor used to fall through to replaying the host's own
    # opening seconds, which renders and scores like a weak failure instead of like a bug.
    if mode != "control" and donor is None:
        raise ValueError(f"run {run.get('name')!r} is a transplant but its donor did not resolve")

    host_states = ann.strided_state(profile, collection, host_ep)
    inject_idx = int(round(t_inject * dp.target_fps))
    start_idx = max(0, inject_idx - pre * chunk)
    t_start = start_idx / dp.target_fps

    host_windows = windows(host_states, start_idx, pre + post, chunk)
    caption_before = run.get("caption_before") or run.get("caption") or ""
    caption_after = run.get("caption") or ""

    program: list[dict] = [{"phase": "grounded", "states": w, "caption": caption_before} for w in host_windows[:pre]]

    if mode == "control":
        program += [{"phase": "grounded", "states": w, "caption": caption_before} for w in host_windows[pre:]]
    else:
        donor_states = ann.strided_state(profile, donor["source"], int(donor["episode_index"]))
        donor_start = int(round(float(donor["t_start"]) * dp.target_fps))
        donor_windows = windows(donor_states, donor_start, post, chunk)
        host_post = continuous(host_windows[pre:], chunk)
        donor_post = continuous(donor_windows, chunk)
        spliced = splice(
            profile,
            host_post,
            donor_post,
            dims=run.get("dims", "all"),
            blend_frames=int(run.get("blend_frames", 4)),
            jaw_amplitude=float(run.get("jaw_amplitude", 1.0)),
            src_arm=donor.get("arm"),
            dst_arm=run.get("arm"),
        )
        program += [{"phase": "failure", "states": w, "caption": caption_after} for w in rechunk(spliced, post, chunk)]

    out_dir.mkdir(parents=True, exist_ok=True)
    initial = out_dir / "initial_obs.png"
    if is_rank0():
        concat_frame(profile, collection, host_ep, t_start).save(initial, format="PNG")
        (out_dir / "record.json").write_text(
            json.dumps(
                {
                    "name": run.get("name"),
                    "recipe": run.get("recipe"),
                    "mode": mode,
                    "host": {
                        "source": collection,
                        "episode_index": host_ep,
                        "t_start": round(t_start, 2),
                        "t_inject": round(t_inject, 2),
                    },
                    "donor": donor,
                    "arm": run.get("arm"),
                    "target_phase": run.get("target_phase"),
                    "dims": run.get("dims") if mode != "control" else None,
                    "blend_frames": run.get("blend_frames") if mode != "control" else None,
                    "jaw_amplitude": run.get("jaw_amplitude") if mode != "control" else None,
                    "pose_distance": run.get("pose_distance"),
                    "pre_chunks": pre,
                    "post_chunks": post,
                    "fps": dp.target_fps,
                    # Seconds into the generated clip, which is what scoring and the grids mark.
                    "failure_time_s": round(t_inject - t_start, 2),
                    "caption": caption_after,
                    "phases": [p["phase"] for p in program],
                },
                indent=2,
            )
        )

    current, clips, encoded = initial, [], []
    for k, step in enumerate(program):
        chunk_dir = out_dir / f"chunk_{k:03d}"
        actions = model.encode_actions(np.asarray(step["states"], dtype=np.float32))
        encoded.append(actions)
        clip = model.rollout_chunk(current, actions, step["caption"], chunk_dir, seed + k)
        nxt = chunk_dir / "last_frame.png"
        if is_rank0():
            last_frame(clip, nxt)
        clips.append(clip)
        current = nxt

    synth = out_dir / "synth.mp4"
    if is_rank0():
        stitch(clips, synth, dp.target_fps)
        # Named for what it holds: the normalized chunks the model was driven with. Keeping them
        # is what lets an export prove its reconstructed trajectories against the real inputs.
        (out_dir / "actions_normalized.json").write_text(json.dumps(encoded))
    return synth
