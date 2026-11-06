#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Visual gripper-state classifier for the LIBERO wrist view, used to check whether the
GENERATED video opens/closes the gripper when the REAL recording does.

train : decode wrist frames of the TRAINING episodes (held-out ids excluded), label them from
        the recorded finger aperture (observation.state[6]-state[7]: open > 0.05, closed < 0.03,
        in-between dropped), train a small CNN on the 160x160 wrist crop taken from the same
        320x192 canvas path the world model uses, and report accuracy on the REAL held-out frames.
apply : run the classifier on gen/real frames saved by fd_rollout_eval.py (frames.npz), smooth,
        extract open<->closed transitions, and match generated transitions to real ones.
"""
import argparse
import glob
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OPEN_THR, CLOSED_THR = 0.05, 0.03  # finger aperture (m); measured: open ~0.079, closed ~0.011-0.03
CANVAS_W, CANVAS_H = 320, 192


def wrist_crop_from_concat_u8(concat_u8_cthw: torch.Tensor) -> torch.Tensor:
    """[3,T,256,512] uint8 concat -> right half of the 160x320 canvas content -> [T,3,160,160] uint8."""
    from cosmos_framework.data.generator.action.utils.transforms import reflection_pad_to_target, remove_reflection_padding

    d: dict = {"video": concat_u8_cthw}
    reflection_pad_to_target(d, ["video"], True, CANVAS_W, CANVAS_H)
    content = remove_reflection_padding(d["video"], d["image_size"])  # [3,T,160,320]
    return content[..., 160:].permute(1, 0, 2, 3).contiguous()


class GripperNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = [3, 16, 32, 64, 96]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1), nn.BatchNorm2d(c[i + 1]), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)  # 160 -> 10
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, 1))

    def forward(self, x):  # x uint8 [B,3,160,160] or float
        x = x.float() / 127.5 - 1.0
        return self.head(self.features(x)).squeeze(1)


_DS = None


def _init_ds(root, holdout, split):
    global _DS
    from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset

    _DS = LIBEROLeRobotDataset(root=root, fps=20, chunk_length=16, mode="forward_dynamics", camera_mode="concat_view",
                               image_size=256, action_space="frame_wise_relative", rotation_space="6d",
                               pose_coordinate_frame="native", action_normalization="quantile_rot", split=split,
                               holdout_episodes_path=holdout)


def _episode_frames(args):
    ep, stride, aperture, command = args
    ds = _DS
    rows = np.flatnonzero(ds._row_episode == ep)
    sel = rows[::stride]
    ts = [float(ds._row_timestamp[j]) for j in sel]
    v = ds._load_video(ds._episodes[int(ep)], ts)  # [n,3,256,512] float [0,1]
    u8 = (v * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3)
    crops = wrist_crop_from_concat_u8(u8).numpy()  # [n,3,160,160]
    # labels are computed on the full-rate rows (exclusion window in frames), then subsampled
    full_lab = labels_from_command(command[rows], np.zeros(len(rows), dtype=np.int64))
    return crops, aperture[sel], np.full(len(sel), ep, dtype=np.int64), full_lab[::stride]


def load_aperture(root: str) -> np.ndarray:
    import pandas as pd

    d = pd.concat([pd.read_parquet(p, columns=["index", "observation.state"])
                   for p in sorted(glob.glob(f"{root}/data/chunk-*/file-*.parquet"))]).sort_values("index")
    S = np.stack(d["observation.state"].values)
    return (S[:, 6] - S[:, 7]).astype(np.float32)


def labels_from_aperture(ap: np.ndarray) -> np.ndarray:
    """1 = open, 0 = closed, -1 = transitional (dropped for training)."""
    lab = np.full(ap.shape, -1, dtype=np.int64)
    lab[ap > OPEN_THR] = 1
    lab[ap < CLOSED_THR] = 0
    return lab


CMD_LAG_EXCLUDE = 10  # frames after a command change whose visual state is ambiguous (actuation lag ~2-9 frames)


def labels_from_command(cmd: np.ndarray, episode: np.ndarray) -> np.ndarray:
    """1 = open (cmd > 0.5), 0 = closed; -1 within CMD_LAG_EXCLUDE frames after a command change (per episode).
    Unlike the aperture rule this calls a gripper closed on a large object 'closed'."""
    state = (cmd > 0.5).astype(np.int64)
    lab = state.copy()
    for e in np.unique(episode):
        idx = np.flatnonzero(episode == e)
        ch = np.flatnonzero(np.diff(state[idx]) != 0) + 1
        for c in ch:
            lab[idx[c : c + CMD_LAG_EXCLUDE]] = -1
    return lab


def load_command(root: str) -> np.ndarray:
    import pandas as pd

    d = pd.concat([pd.read_parquet(p, columns=["index", "action"])
                   for p in sorted(glob.glob(f"{root}/data/chunk-*/file-*.parquet"))]).sort_values("index")
    return np.stack(d["action"].values)[:, 6].astype(np.float32)


def cmd_train(a):
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    aperture = load_aperture(a.root)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    command = load_command(a.root)

    def collect(split, stride):
        _init_ds(a.root, a.holdout, split)
        eps = [int(e) for e in _DS._ep_vals.tolist()]
        with Pool(a.workers, initializer=_init_ds, initargs=(a.root, a.holdout, split)) as pool:
            parts = pool.map(_episode_frames, [(e, stride, aperture, command) for e in eps])
        X = np.concatenate([p[0] for p in parts]); AP = np.concatenate([p[1] for p in parts]); E = np.concatenate([p[2] for p in parts])
        CL = np.concatenate([p[3] for p in parts])
        return X, AP, E, CL, eps

    Xtr, APtr, Etr, CLtr, eps_tr = collect("train", a.stride)
    Xva, APva, Eva, CLva, eps_va = collect("val", 1)
    if a.label_source == "cmd":
        ytr, yva = CLtr, CLva
    else:
        ytr, yva = labels_from_aperture(APtr), labels_from_aperture(APva)
    ktr, kva = ytr >= 0, yva >= 0
    print(f"[train] train frames {len(Xtr)} (labeled {ktr.sum()}, open {int((ytr==1).sum())}, closed {int((ytr==0).sum())}) "
          f"from {len(eps_tr)} episodes; val frames {len(Xva)} (labeled {kva.sum()}) from {len(eps_va)} held-out episodes", flush=True)
    Xtr_t = torch.from_numpy(Xtr[ktr]); ytr_t = torch.from_numpy(ytr[ktr]).float()
    Xva_t = torch.from_numpy(Xva); yva_np = yva
    net = GripperNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    n = len(Xtr_t); bs = 128; steps = math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * steps)
    g = torch.Generator().manual_seed(0)
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(n, generator=g); tot = 0.0
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb = Xtr_t[idx].to(dev, non_blocking=True); yb = ytr_t[idx].to(dev)
            # light augmentation: random brightness/contrast jitter + small translation
            if a.augment:
                xb = xb.float()
                gain = torch.empty(len(xb), 1, 1, 1, device=dev).uniform_(0.8, 1.2)
                bias = torch.empty(len(xb), 1, 1, 1, device=dev).uniform_(-20, 20)
                xb = (xb * gain + bias).clamp(0, 255)
                dx, dy = np.random.randint(-6, 7, size=2)
                xb = torch.roll(xb, shifts=(int(dy), int(dx)), dims=(2, 3))
            loss = F.binary_cross_entropy_with_logits(net(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        net.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(net(Xva_t[i:i + 512].to(dev))).cpu() for i in range(0, len(Xva_t), 512)]).numpy()
        pred = (p > 0.5).astype(int)
        acc = float((pred[kva] == yva_np[kva]).mean())
        print(f"[train] epoch {ep + 1}/{a.epochs} loss {tot / steps:.4f} | held-out REAL frames: acc {acc:.4f} on {int(kva.sum())} labeled frames", flush=True)
    # per-episode accuracy on held-out real frames + agreement with the aperture label including transitional frames
    per_ep = {}
    for e in eps_va:
        m = (Eva == e) & kva
        per_ep[int(e)] = float((pred[m] == yva_np[m]).mean()) if m.any() else None
    torch.save({"state_dict": net.state_dict(), "open_thr": OPEN_THR, "closed_thr": CLOSED_THR, "label_source": a.label_source}, out / "gripper_net.pt")
    rep = dict(label_source=a.label_source, train_frames=int(ktr.sum()), train_episodes=len(eps_tr), heldout_frames_labeled=int(kva.sum()),
               heldout_frames_total=int(len(Xva)), heldout_real_accuracy=acc, per_episode_accuracy=per_ep,
               epochs=a.epochs, stride=a.stride, thresholds=dict(open=OPEN_THR, closed=CLOSED_THR, cmd_lag_exclude=CMD_LAG_EXCLUDE))
    (out / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("train_frames", "heldout_frames_labeled", "heldout_real_accuracy")}), flush=True)


def smooth_states(prob: np.ndarray, win: int = 5, hi: float = 0.6, lo: float = 0.4) -> np.ndarray:
    """median filter + hysteresis -> 0/1 state sequence."""
    from scipy.ndimage import median_filter

    p = median_filter(prob, size=win, mode="nearest")
    s = np.zeros(len(p), dtype=int); cur = int(p[0] > 0.5)
    for i, v in enumerate(p):
        if cur == 1 and v < lo:
            cur = 0
        elif cur == 0 and v > hi:
            cur = 1
        s[i] = cur
    return s


def transitions(s: np.ndarray) -> list[tuple[int, str]]:
    ev = []
    for t in range(1, len(s)):
        if s[t] != s[t - 1]:
            ev.append((t, "close" if s[t] == 0 else "open"))
    return ev


def match_events(real_ev, gen_ev, tol: int):
    used = set(); matched = []
    for t_r, kind in real_ev:
        best = None
        for j, (t_g, kg) in enumerate(gen_ev):
            if j in used or kg != kind:
                continue
            if abs(t_g - t_r) <= tol and (best is None or abs(t_g - t_r) < abs(gen_ev[best][0] - t_r)):
                best = j
        if best is not None:
            used.add(best); matched.append((t_r, gen_ev[best][0], kind))
    false_alarms = [e for j, e in enumerate(gen_ev) if j not in used]
    return matched, false_alarms


def cmd_apply(a):
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    ck = torch.load(a.model, map_location="cpu")
    net = GripperNet(); net.load_state_dict(ck["state_dict"]); net.to(dev).eval()
    ev_dir = Path(a.eval_dir)
    results = []
    for npz in sorted(ev_dir.glob("episodes/ep*/frames.npz")):
        d = np.load(npz)
        gen, real = d["gen"], d["real"]  # [T,3,160,320]
        with torch.no_grad():
            def run(x):
                x = torch.from_numpy(np.ascontiguousarray(x[..., 160:]))
                return torch.cat([torch.sigmoid(net(x[i:i + 512].to(dev))).cpu() for i in range(0, len(x), 512)]).numpy()
            p_gen, p_real = run(gen), run(real)
        s_gen, s_real = smooth_states(p_gen), smooth_states(p_real)
        ev_real, ev_gen = transitions(s_real), transitions(s_gen)
        matched, fa = match_events(ev_real, ev_gen, a.tol)
        cmd = d["actions10"][:, 9] if "actions10" in d else None  # raw gripper command (1 open, 0 closed) per frame transition
        cmd_ev = []
        if cmd is not None:
            cs = (cmd > 0.5).astype(int)
            for t in np.flatnonzero(np.diff(cs) != 0) + 1:
                cmd_ev.append((int(t), "close" if cs[t] == 0 else "open"))
        # generated event matched to a command change if it follows it within [0, tol_lag] frames
        def match_to_cmd(ev):
            used = set(); m = []
            for t_c, k in cmd_ev:
                best = None
                for j, (t_g, kg) in enumerate(ev):
                    if j in used or kg != k or t_g < t_c or t_g - t_c > a.tol_lag:
                        continue
                    if best is None or t_g < ev[best][0]:
                        best = j
                if best is not None:
                    used.add(best); m.append((t_c, ev[best][0], k))
            return m
        m_gen_cmd, m_real_cmd = match_to_cmd(ev_gen), match_to_cmd(ev_real)
        res = dict(
            episode=int(npz.parent.name[2:]), frames=int(len(gen)),
            frame_agreement=float((s_gen[1:] == s_real[1:]).mean()),
            real_transitions=[(int(t), k) for t, k in ev_real], gen_transitions=[(int(t), k) for t, k in ev_gen],
            matched=[(int(tr), int(tg), k) for tr, tg, k in matched],
            n_real=len(ev_real), n_gen=len(ev_gen), n_matched=len(matched), n_false_alarm=len(fa),
            mean_abs_dt_frames=float(np.mean([abs(tg - tr) for tr, tg, _ in matched])) if matched else None,
            cmd_transitions=[int(t) for t, _ in cmd_ev],
            n_cmd=len(cmd_ev), n_gen_matched_cmd=len(m_gen_cmd), n_real_matched_cmd=len(m_real_cmd),
            gen_lag_after_cmd=[int(tg - tc) for tc, tg, _ in m_gen_cmd], real_lag_after_cmd=[int(tg - tc) for tc, tg, _ in m_real_cmd],
            p_gen=p_gen.round(3).tolist(), p_real=p_real.round(3).tolist(),
        )
        results.append(res)
        print(f"[apply] ep {res['episode']}: real events {res['n_real']} gen events {res['n_gen']} matched {res['n_matched']} "
              f"false alarms {res['n_false_alarm']} frame-agreement {res['frame_agreement']:.3f} "
              f"mean|dt| {res['mean_abs_dt_frames']}", flush=True)
    n_real = sum(r["n_real"] for r in results); n_matched = sum(r["n_matched"] for r in results)
    n_gen = sum(r["n_gen"] for r in results)
    dts = [abs(tg - tr) for r in results for tr, tg, _ in r["matched"]]
    n_cmd = sum(r["n_cmd"] for r in results); n_gc = sum(r["n_gen_matched_cmd"] for r in results); n_rc = sum(r["n_real_matched_cmd"] for r in results)
    lag_g = [x for r in results for x in r["gen_lag_after_cmd"]]; lag_r = [x for r in results for x in r["real_lag_after_cmd"]]
    agg = dict(
        tolerance_frames=a.tol, tol_lag_frames=a.tol_lag, n_episodes=len(results),
        command_transitions=n_cmd,
        real_video_events_matched_to_command=n_rc, real_vs_command_recall=(n_rc / n_cmd) if n_cmd else None,
        gen_video_events_matched_to_command=n_gc, gen_vs_command_recall=(n_gc / n_cmd) if n_cmd else None,
        mean_lag_frames_real=float(np.mean(lag_r)) if lag_r else None, mean_lag_frames_gen=float(np.mean(lag_g)) if lag_g else None,
        real_transitions=n_real, gen_transitions=n_gen, matched=n_matched,
        event_recall=(n_matched / n_real) if n_real else None, event_precision=(n_matched / n_gen) if n_gen else None,
        mean_abs_dt_frames=float(np.mean(dts)) if dts else None, median_abs_dt_frames=float(np.median(dts)) if dts else None,
        frame_agreement_mean=float(np.mean([r["frame_agreement"] for r in results])),
    )
    out = ev_dir / "gripper_check.json"
    out.write_text(json.dumps(dict(aggregate=agg, per_episode=results), indent=1))
    print(json.dumps(agg, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--root", required=True); t.add_argument("--holdout", required=True)
    t.add_argument("--out", required=True); t.add_argument("--stride", type=int, default=4); t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--workers", type=int, default=16); t.add_argument("--cpu", action="store_true"); t.add_argument("--augment", action="store_true", default=True)
    p = sub.add_parser("apply"); p.add_argument("--model", required=True); p.add_argument("--eval-dir", required=True)
    p.add_argument("--tol", type=int, default=10, help="matching tolerance in frames (10 = 0.5 s at 20 FPS)"); p.add_argument("--cpu", action="store_true")
    p.add_argument("--tol-lag", type=int, default=20, help="a video event matches a command change if it follows it within this many frames")
    t.add_argument("--label-source", default="cmd", choices=["cmd", "aperture"], help="training labels: gripper command (default) or finger aperture")
    a = ap.parse_args()
    {"train": cmd_train, "apply": cmd_apply}[a.cmd](a)
