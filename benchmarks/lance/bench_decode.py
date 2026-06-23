# SPDX-License-Identifier: OpenMDW-1.1
"""Microbenchmark isolating the action loader's bottleneck: multi-view video
decode. Strips the shared tabular/pose work so the numbers reflect only how
fast each backend turns (episode, timestamps) into frames.

  base       — lerobot decode_video_frames(mp4 path, ts) per view (CPU torchcodec,
               decoder cached by path — the base loader's exact path)
  lance-cpu  — VideoDecoder(blob) CPU, batched get_frames_at across the window set
  lance-gpu  — VideoDecoder(blob) NVDEC, batched

Reports decoded video-frames/sec (3 views × (chunk+1) frames per window).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def _windows(base_ds, k, seed=0):
    rng = np.random.RandomState(seed)
    idxs = rng.randint(0, len(base_ds), size=k)
    out = []
    for idx in idxs:
        ep = int(np.searchsorted(base_ds._valid_cum, idx, side="right"))
        prev = int(base_ds._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(base_ds._ep_starts[ep]) + (int(idx) - prev)
        episode_index = int(base_ds._ep_vals[ep])
        episode = base_ds._episodes[episode_index]
        obs = base_ds._window_rows(start, start + base_ds._chunk_length + 1, episode_index)
        # global row range for the window (lance row id == global frame order)
        out.append((episode, [float(r["timestamp"]) for r in obs], start))
    return out


def _bench_base(root, windows, repeat):
    """Base loader's exact path: lerobot decode_video_frames (CPU torchcodec,
    decoder cached by path)."""
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import _IMAGE_FEATURES
    from lerobot.datasets.video_utils import decode_video_frames

    import json
    from pathlib import Path

    info = json.loads((Path(root) / "meta" / "info.json").read_text())

    def vp(ep, vk):
        ci = int(ep.get(f"videos/{vk}/chunk_index", 0))
        fi = int(ep.get(f"videos/{vk}/file_index", 0))
        return Path(root) / info["video_path"].format(video_key=vk, chunk_index=ci, file_index=fi)

    # warmup (prime decoder cache + page cache)
    for episode, ts, _s in windows[: min(8, len(windows))]:
        for _n, vk in _IMAGE_FEATURES.items():
            from_ts = float(episode.get(f"videos/{vk}/from_timestamp", 0.0))
            decode_video_frames(vp(episode, vk), [from_ts + t for t in ts], 2e-4)
    t0 = time.perf_counter()
    nframes = 0
    for _ in range(repeat):
        for episode, ts, _s in windows:
            for _n, vk in _IMAGE_FEATURES.items():
                from_ts = float(episode.get(f"videos/{vk}/from_timestamp", 0.0))
                f = decode_video_frames(vp(episode, vk), [from_ts + t for t in ts], 2e-4)
                nframes += f.shape[0]
    return nframes / (time.perf_counter() - t0)


def _bench_base_gpu(root, windows, repeat):
    """Fair control: plain mp4 FILES decoded on the GPU (NVDEC), same batched
    get_frames_at as the lance path. Isolates 'NVDEC' from 'lance storage'."""
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import _IMAGE_FEATURES
    from torchcodec.decoders import VideoDecoder
    import json
    from pathlib import Path

    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    decoders: dict[str, VideoDecoder] = {}

    def dec_for(vk, ep):
        ci = int(ep.get(f"videos/{vk}/chunk_index", 0))
        fi = int(ep.get(f"videos/{vk}/file_index", 0))
        path = str(Path(root) / info["video_path"].format(video_key=vk, chunk_index=ci, file_index=fi))
        d = decoders.get(path)
        if d is None:
            d = VideoDecoder(path, device="cuda")
            decoders[path] = d
        return d

    def decode_all(ws):
        plan = {}
        for episode, ts, _s in ws:
            for _n, vk in _IMAGE_FEATURES.items():
                d = dec_for(vk, episode)
                avg = d.metadata.average_fps
                from_ts = float(episode.get(f"videos/{vk}/from_timestamp", 0.0))
                plan.setdefault(id(d), (d, []))[1].extend(round((from_ts + t) * avg) for t in ts)
        nf = 0
        for _k, (d, fidx) in plan.items():
            nf += d.get_frames_at(indices=fidx).data.shape[0]
        torch.cuda.synchronize()
        return nf

    decode_all(windows[: min(8, len(windows))])
    t0 = time.perf_counter()
    nframes = sum(decode_all(windows) for _ in range(repeat))
    return nframes / (time.perf_counter() - t0)


def _bench_lance(root, uri, windows, repeat, device):
    from cosmos_framework.data.lance import LanceDROIDDataset
    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import _IMAGE_FEATURES

    ds = LanceDROIDDataset(
        root=root, lance_uri=uri, decode_device=device,
        action_space="joint_pos", use_state=True, mode="policy", chunk_length=16,
    )
    ds._ensure_lance_open()

    def decode_all(windows):
        plan = {}
        for episode, ts, _s in windows:
            for _n, vk in _IMAGE_FEATURES.items():
                ci, fi = ds._video_chunk_file(episode, vk)
                dec = ds._decoder_for(vk, ci, fi)
                avg = dec.metadata.average_fps
                from_ts = float(episode.get(f"videos/{vk}/from_timestamp", 0.0))
                fidx = [round((from_ts + t) * avg) for t in ts]
                plan.setdefault((vk, ci, fi), []).extend(fidx)
        nf = 0
        for key, fidx in plan.items():
            out = ds._decoder_for(*key).get_frames_at(indices=fidx)
            nf += out.data.shape[0]
        if device == "cuda":
            torch.cuda.synchronize()
        return nf

    decode_all(windows[: min(8, len(windows))])  # warmup
    t0 = time.perf_counter()
    nframes = 0
    for _ in range(repeat):
        nframes += decode_all(windows)
    return nframes / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--uri", required=True, help="video-blob lance dir")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

    base_ds = DROIDLeRobotDataset(
        root=args.root, action_space="joint_pos", use_state=True, mode="policy", chunk_length=16
    )
    windows = _windows(base_ds, args.windows)
    print(f"{args.windows} windows × {args.repeat} repeats, 3 views × {base_ds._chunk_length + 1} frames each\n")

    base = _bench_base(args.root, windows, args.repeat)
    bgpu = _bench_base_gpu(args.root, windows, args.repeat)
    lcpu = _bench_lance(args.root, args.uri, windows, args.repeat, "cpu")
    lgpu = _bench_lance(args.root, args.uri, windows, args.repeat, "cuda")
    rows = [
        ("base-cpu", "mp4 file", "CPU (h264)", base),
        ("base-gpu", "mp4 file", "NVDEC", bgpu),
        ("lance-video-cpu", "blob-v2", "CPU (h264)", lcpu),
        ("lance-video-gpu", "blob-v2", "NVDEC", lgpu),
    ]
    print(f"\n{'backend':<18}{'storage':>10}{'decode':>14}{'frames/s':>12}{'vs base':>10}")
    for name, store, dec, v in rows:
        print(f"{name:<18}{store:>10}{dec:>14}{v:>12.0f}{v / base:>9.2f}x")


if __name__ == "__main__":
    main()
